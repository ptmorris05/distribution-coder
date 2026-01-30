use numpy::{PyReadonlyArray1, Element, PyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};
use half::{f16, bf16}; 

// --- CONFIGURATION ---
const PRECISION: u64 = 62;
const MAX_VAL: u64 = (1 << PRECISION) - 1;
const QUARTER: u64 = 1 << (PRECISION - 2);
const HALF: u64 = 1 << (PRECISION - 1);
const THREE_QUARTER: u64 = QUARTER * 3;

const FREQ_BITS: u64 = 32;
const TOTAL_FREQ: u64 = 1 << FREQ_BITS;

#[pyclass]
struct ArithmeticCoder {
    low: u64,
    high: u64,
    follow_bits: u64,
    buffer: Vec<u8>,
    buffer_bit_count: u8,
    buffer_byte: u8,

    value: u64,
    input_buffer: Vec<u8>,
    input_cursor: usize,
    input_bit_count: u8,
}

#[pymethods]
impl ArithmeticCoder {
    #[new]
    fn new() -> Self {
        ArithmeticCoder {
            low: 0,
            high: MAX_VAL,
            follow_bits: 0,
            buffer: Vec::with_capacity(4096),
            buffer_bit_count: 0,
            buffer_byte: 0,
            
            value: 0,
            input_buffer: Vec::new(),
            input_cursor: 0,
            input_bit_count: 0,
        }
    }

    fn start_decoding(&mut self, input_bytes: &[u8]) {
        self.input_buffer = input_bytes.to_vec();
        self.input_cursor = 0;
        self.input_bit_count = 0; 
        self.low = 0;
        self.high = MAX_VAL;
        self.value = 0;

        for _ in 0..PRECISION {
            self.value = (self.value << 1) | self.read_bit();
        }
    }

    // --- ENCODE DISPATCHER ---
    fn encode_step(&mut self, probs: &Bound<'_, PyAny>, symbol: usize) -> PyResult<()> {
        if let Ok(arr) = probs.downcast::<PyArray1<f32>>() {
            return self.encode_step_generic(arr.readonly(), symbol);
        }
        if let Ok(arr) = probs.downcast::<PyArray1<f16>>() {
            return self.encode_step_generic(arr.readonly(), symbol);
        }
        if let Ok(arr) = probs.downcast::<PyArray1<bf16>>() {
            return self.encode_step_generic(arr.readonly(), symbol);
        }
        if let Ok(arr) = probs.downcast::<PyArray1<f64>>() {
            return self.encode_step_generic(arr.readonly(), symbol);
        }

        // Fallback: Python-side cast to f32
        let py = probs.py();
        let numpy_mod = py.import_bound("numpy")?;
        let arr_f32 = numpy_mod.call_method1("array", (probs, "float32"))?
                               .downcast_into::<PyArray1<f32>>()?;
        
        self.encode_step_generic(arr_f32.readonly(), symbol)
    }

    // --- DECODE DISPATCHER ---
    fn decode_step(&mut self, probs: &Bound<'_, PyAny>) -> PyResult<usize> {
        if let Ok(arr) = probs.downcast::<PyArray1<f32>>() {
            return Ok(self.decode_step_generic(arr.readonly()));
        }
        if let Ok(arr) = probs.downcast::<PyArray1<f16>>() {
            return Ok(self.decode_step_generic(arr.readonly()));
        }
        if let Ok(arr) = probs.downcast::<PyArray1<bf16>>() {
            return Ok(self.decode_step_generic(arr.readonly()));
        }
        if let Ok(arr) = probs.downcast::<PyArray1<f64>>() {
            return Ok(self.decode_step_generic(arr.readonly()));
        }

        let py = probs.py();
        let numpy_mod = py.import_bound("numpy")?;
        let arr_f32 = numpy_mod.call_method1("array", (probs, "float32"))?
                               .downcast_into::<PyArray1<f32>>()?;
        
        Ok(self.decode_step_generic(arr_f32.readonly()))
    }

    fn finish_encoding<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        self.follow_bits += 1;
        if self.low < QUARTER {
            self.emit_bit_plus_follow(0);
        } else {
            self.emit_bit_plus_follow(1);
        }
        
        if self.buffer_bit_count > 0 {
            self.buffer.push(self.buffer_byte << (8 - self.buffer_bit_count));
        }
        PyBytes::new(py, &self.buffer)
    }
}

// --- GENERIC IMPLEMENTATION ---
impl ArithmeticCoder {

    // TRAIT BOUNDS: T must be Copy, an Element (Numpy), and convertible to f64
    fn encode_step_generic<T>(&mut self, probs: PyReadonlyArray1<T>, symbol: usize) -> PyResult<()> 
    where T: Element + Copy + Into<f64> 
    {
        let probs_slice = probs.as_slice().unwrap();
        let (max_idx, diff) = self.analyze_probs_generic(probs_slice);

        let mut sym_low: u64 = 0;
        let mut sym_high: u64 = 0;
        let mut running_sum: u64 = 0;

        for (i, &p) in probs_slice.iter().enumerate() {
            let val_f64: f64 = p.into();
            let f = (val_f64 * TOTAL_FREQ as f64) as u64;
            let mut freq = if f == 0 { 1 } else { f };

            if i == max_idx {
                if diff > 0 { freq += diff as u64; }
                else { freq -= (-diff) as u64; }
            }

            let next_sum = running_sum + freq;

            if i == symbol {
                sym_low = running_sum;
                sym_high = next_sum;
                break;
            }
            running_sum = next_sum;
        }

        let range = self.high - self.low + 1;
        let range_128 = range as u128;
        let dist_high = (range_128 * sym_high as u128) / TOTAL_FREQ as u128;
        let dist_low  = (range_128 * sym_low as u128) / TOTAL_FREQ as u128;

        self.high = self.low + (dist_high as u64) - 1;
        self.low = self.low + (dist_low as u64);

        self.renormalize_encode();
        Ok(())
    }

    fn decode_step_generic<T>(&mut self, probs: PyReadonlyArray1<T>) -> usize 
    where T: Element + Copy + Into<f64>
    {
        let probs_slice = probs.as_slice().unwrap();
        let (max_idx, diff) = self.analyze_probs_generic(probs_slice);

        let range = self.high - self.low + 1;
        let offset = self.value - self.low;
        let numerator = (offset as u128 * TOTAL_FREQ as u128).saturating_sub(1);
        let target_count = (numerator / range as u128) as u64;

        let mut symbol = 0;
        let mut sym_low = 0;
        let mut sym_high = 0;
        let mut running_sum = 0;

        for (i, &p) in probs_slice.iter().enumerate() {
            let val_f64: f64 = p.into();
            let f = (val_f64 * TOTAL_FREQ as f64) as u64;
            let mut freq = if f == 0 { 1 } else { f };

            if i == max_idx {
                if diff > 0 { freq += diff as u64; }
                else { freq -= (-diff) as u64; }
            }

            let next_sum = running_sum + freq;

            if target_count < next_sum {
                symbol = i;
                sym_low = running_sum;
                sym_high = next_sum;
                break;
            }
            running_sum = next_sum;
        }

        let range_128 = range as u128;
        let dist_high = (range_128 * sym_high as u128) / TOTAL_FREQ as u128;
        let dist_low  = (range_128 * sym_low as u128) / TOTAL_FREQ as u128;

        self.high = self.low + (dist_high as u64) - 1;
        self.low = self.low + (dist_low as u64);

        self.renormalize_decode();
        symbol
    }

    #[inline(always)]
    fn analyze_probs_generic<T>(&self, probs: &[T]) -> (usize, i64) 
    where T: Copy + Into<f64>
    {
        let mut current_sum: u64 = 0;
        let mut max_val: u64 = 0;
        let mut max_idx: usize = 0;

        for (i, &p) in probs.iter().enumerate() {
            let val_f64: f64 = p.into();
            let f = (val_f64 * TOTAL_FREQ as f64) as u64;
            let freq = if f == 0 { 1 } else { f };

            if freq > max_val {
                max_val = freq;
                max_idx = i;
            }
            current_sum += freq;
        }
        let diff = TOTAL_FREQ as i64 - current_sum as i64;
        (max_idx, diff)
    }

    #[inline(always)]
    fn renormalize_encode(&mut self) {
        loop {
            if self.high < HALF {
                self.emit_bit_plus_follow(0);
            } else if self.low >= HALF {
                self.emit_bit_plus_follow(1);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.follow_bits += 1;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low *= 2;
            self.high = self.high * 2 + 1;
        }
    }

    #[inline(always)]
    fn renormalize_decode(&mut self) {
        loop {
            if self.high < HALF {
            } else if self.low >= HALF {
                self.value -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= QUARTER && self.high < THREE_QUARTER {
                self.value -= QUARTER;
                self.low -= QUARTER;
                self.high -= QUARTER;
            } else {
                break;
            }
            self.low *= 2;
            self.high = self.high * 2 + 1;
            self.value = (self.value << 1) | self.read_bit();
        }
    }

    fn emit_bit_plus_follow(&mut self, bit: u8) {
        self.emit_bit(bit);
        let inv_bit = 1 - bit;
        while self.follow_bits > 0 {
            self.emit_bit(inv_bit);
            self.follow_bits -= 1;
        }
    }

    #[inline(always)]
    fn emit_bit(&mut self, bit: u8) {
        self.buffer_byte = (self.buffer_byte << 1) | bit;
        self.buffer_bit_count += 1;
        if self.buffer_bit_count == 8 {
            self.buffer.push(self.buffer_byte);
            self.buffer_bit_count = 0;
            self.buffer_byte = 0;
        }
    }

    #[inline(always)]
    fn read_bit(&mut self) -> u64 {
        if self.input_cursor >= self.input_buffer.len() && self.input_bit_count == 0 {
            return 0; 
        }
        
        if self.input_bit_count == 0 {
            self.input_bit_count = 8;
            self.input_cursor += 1;
        }
        
        let byte_idx = self.input_cursor - 1;
        if byte_idx >= self.input_buffer.len() { return 0; }

        let bit = (self.input_buffer[byte_idx] >> (self.input_bit_count - 1)) & 1;
        self.input_bit_count -= 1;
        bit as u64
    }
}

#[pymodule]
fn _distribution_coder(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<ArithmeticCoder>()?;
    Ok(())
}