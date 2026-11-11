use anyhow::{bail, ensure, Context, Result};
use libloading::Library;
use std::{ffi::{c_char,c_int}, fs::File, io::{BufWriter,Write}, path::Path};

// NumPy's bundled OpenBLAS is ILP64: matrix dimensions and return values are i64.
type Gesdd = unsafe extern "C" fn(c_int,c_char,i64,i64,*mut f64,i64,*mut f64,*mut f64,i64,*mut f64,i64)->i64;
type Gemm = unsafe extern "C" fn(c_int,c_int,c_int,i64,i64,i64,f64,*const f64,i64,*const f64,i64,f64,*mut f64,i64);

pub struct Backend { _lib: Library, svd: Gesdd, gemm: Gemm }
pub struct Svd { pub m:usize, pub n:usize, pub rank:usize, pub u:Vec<f64>, pub s:Vec<f64>, pub vt:Vec<f64> }

impl Backend {
    pub fn new()->Result<Self> {
        let path=std::env::var("WEEK10_OPENBLAS").context("WEEK10_OPENBLAS is unset; run: source scripts/env.sh")?;
        unsafe {
            let lib=Library::new(&path).with_context(||format!("loading OpenBLAS {path}"))?;
            let svd=*lib.get::<Gesdd>(b"scipy_LAPACKE_dgesdd64_\0")?;
            let gemm=*lib.get::<Gemm>(b"scipy_cblas_dgemm64_\0")?;
            Ok(Self{_lib:lib,svd,gemm})
        }
    }
    pub fn svd(&self, data:&[f64], m:usize,n:usize)->Result<Svd> {
        ensure!(m>0 && n>0 && data.len()==m*n,"invalid SVD dimensions");
        let rank=m.min(n);let mut a=data.to_vec();let mut s=vec![0.;rank];
        let mut u=vec![0.;m*rank];let mut vt=vec![0.;rank*n];
        let info=unsafe{(self.svd)(101,b'S' as c_char,m as i64,n as i64,a.as_mut_ptr(),n as i64,
                                  s.as_mut_ptr(),u.as_mut_ptr(),rank as i64,vt.as_mut_ptr(),n as i64)};
        if info!=0 {bail!("LAPACKE dgesdd failed with info={info} for {m}x{n}")}
        ensure!(s.iter().all(|x|x.is_finite() && *x>=0.),"non-finite SVD spectrum");
        Ok(Svd{m,n,rank,u,s,vt})
    }
    pub fn multiply(&self,a:&[f64],b:&[f64],m:usize,k:usize,n:usize)->Result<Vec<f64>> {
        ensure!(a.len()==m*k&&b.len()==k*n&&m>0&&n>0&&k>0,"invalid matrix multiplication shape");
        let mut c=vec![0.;m*n];
        unsafe{(self.gemm)(101,111,111,m as i64,n as i64,k as i64,1.,a.as_ptr(),k as i64,
                           b.as_ptr(),n as i64,0.,c.as_mut_ptr(),n as i64)}
        Ok(c)
    }
}

pub fn write_npy(path:&Path, dims:&[usize], data:&[f64])->Result<()> {
    ensure!(dims.iter().product::<usize>()==data.len(),"npy shape mismatch");
    let tuple=if dims.len()==1 {format!("{},",dims[0])} else {dims.iter().map(|x|x.to_string()).collect::<Vec<_>>().join(", ")};
    let mut header=format!("{{'descr': '<f8', 'fortran_order': False, 'shape': ({tuple}), }}");
    let pad=(16-(10+header.len()+1)%16)%16;header.push_str(&" ".repeat(pad));header.push('\n');
    ensure!(header.len()<=u16::MAX as usize,"npy header too long");
    let mut f=BufWriter::new(File::create(path)?);f.write_all(b"\x93NUMPY")?;f.write_all(&[1,0])?;
    f.write_all(&(header.len() as u16).to_le_bytes())?;f.write_all(header.as_bytes())?;
    for &x in data {f.write_all(&x.to_le_bytes())?} f.flush()?;Ok(())
}
