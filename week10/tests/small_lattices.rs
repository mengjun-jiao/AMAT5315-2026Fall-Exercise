#[path="../src/linalg.rs"] mod linalg;
#[path="../src/boundary.rs"] mod boundary;
#[path="../src/trg.rs"] mod trg;

use linalg::Backend;
use std::path::Path;

fn exact(l:usize,k:f64,periodic:bool)->(f64,f64) {
    let n=l*l;let mut z=0.;let mut hsum=0.;let mut h2sum=0.;
    for bits in 0..(1usize<<n) {
        let spin=|i:usize|if (bits>>i)&1==0{-1.}else{1.};
        let mut energy=0.;
        for row in 0..l {for col in 0..l {
            let i=row*l+col;
            if periodic||col+1<l {energy-=spin(i)*spin(row*l+(col+1)%l)}
            if periodic||row+1<l {energy-=spin(i)*spin(((row+1)%l)*l+col)}
        }}
        let weight=(-k*energy).exp();z+=weight;hsum+=weight*energy;h2sum+=weight*energy*energy;
    }
    let mean=hsum/z;
    (z.ln()/n as f64,k*k*(h2sum/z-mean*mean)/n as f64)
}

#[test]
fn open_boundary_matches_all_spin_assignments() {
    let backend=Backend::new().unwrap();
    for k in [0.,0.44,0.7] {
        let got=boundary::solve(&backend,4,k,8,false).unwrap().log_z/16.;
        assert!((got-exact(4,k,false).0).abs()<1e-8);
    }
    let k=0.44;let h=0.0025;
    let g=|x:f64|boundary::solve(&backend,4,x,8,false).unwrap().log_z/16.;
    let cv=k*k*(g(k+h)-2.*g(k)+g(k-h))/(h*h);
    assert!((cv-exact(4,k,false).1).abs()<1e-3);
}

#[test]
fn periodic_closure_and_double_bonds_match_enumeration() {
    let backend=Backend::new().unwrap();let path=Path::new(".");
    for (l,chi,tol) in [(2,16,1e-8),(4,8,2e-3)] {
        let got=trg::solve(&backend,l,0.44,chi,false,path).unwrap().log_z/(l*l) as f64;
        assert!((got-exact(l,0.44,true).0).abs()<tol);
    }
    let k=0.44;let h=0.0025;
    let g=|x:f64|trg::solve(&backend,2,x,16,false,path).unwrap().log_z/4.;
    let cv=k*k*(g(k+h)-2.*g(k)+g(k-h))/(h*h);
    assert!((cv-exact(2,k,true).1).abs()<1e-3);
}
