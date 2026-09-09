//! Pure pair physics shared by simulation and trajectory verification.
#[derive(Clone, Copy, Debug)]
pub enum PhysicalModel {
    OpenLennardJones,
    PeriodicShiftedLennardJones { box_size: [f64; 2], rc: f64 },
}
impl PhysicalModel {
    pub fn validate(&self) -> Result<(), String> {
        if let Self::PeriodicShiftedLennardJones { box_size, rc } = self {
            if !rc.is_finite() || *rc <= 0.0 || box_size.iter().any(|l| !l.is_finite() || *l <= 2.0*rc) {
                return Err("box lengths must be finite and greater than 2*rc".into());
            }
        }
        Ok(())
    }
    pub fn displacement(&self, a:[f64;2], b:[f64;2])->[f64;2] {
        let mut d=[a[0]-b[0],a[1]-b[1]];
        if let Self::PeriodicShiftedLennardJones { box_size, .. }=self {
            for axis in 0..2 { d[axis]-=box_size[axis]*(d[axis]/box_size[axis]).round(); }
        }
        d
    }
    pub fn wrap(&self, mut p:[f64;2])->[f64;2] {
        if let Self::PeriodicShiftedLennardJones { box_size, .. }=self {
            for axis in 0..2 {
                p[axis]=p[axis].rem_euclid(box_size[axis]);
                if p[axis]==box_size[axis] { p[axis]=0.0; }
            }
        }
        p
    }
    pub fn pair(&self, r:f64)->Result<(f64,f64),String> {
        if !r.is_finite() || r<=0.0 { return Err("nonzero finite separation required".into()); }
        let u=match self {
            Self::OpenLennardJones=>crate::energy(r),
            Self::PeriodicShiftedLennardJones {rc,..}=> {
                if r>=*rc { return Ok((0.,0.)); }
                crate::energy(r)-crate::energy(*rc)
            }
        };
        let f=crate::force(r);
        if !u.is_finite() || !f.is_finite() { return Err("nonfinite pair energy or force".into()); }
        Ok((u,f))
    }
}
fn validate_positions(pos:&[[f64;2]],model:&PhysicalModel)->Result<(),String> {
    model.validate()?;
    if pos.len()<2 || !pos.iter().flatten().all(|x|x.is_finite()) { return Err("at least two finite positions required".into()); }
    Ok(())
}
pub fn accelerations(pos:&[[f64;2]],model:&PhysicalModel)->Result<Vec<[f64;2]>,String> {
    validate_positions(pos,model)?;
    let mut a=vec![[0.;2];pos.len()];
    for i in 0..pos.len() { for j in i+1..pos.len() {
        let d=model.displacement(pos[i],pos[j]); let r=d[0].hypot(d[1]);
        let (_,f)=model.pair(r)?;
        for axis in 0..2 { let component=f*d[axis]/r; a[i][axis]+=component; a[j][axis]-=component; }
    }}
    if !a.iter().flatten().all(|x|x.is_finite()) { return Err("nonfinite acceleration".into()); }
    Ok(a)
}
pub fn energies(pos:&[[f64;2]],vel:&[[f64;2]],model:&PhysicalModel)->Result<(f64,f64),String> {
    validate_positions(pos,model)?;
    if vel.len()!=pos.len() || !vel.iter().flatten().all(|x|x.is_finite()) { return Err("matching finite velocities required".into()); }
    let mut u=0.;
    for i in 0..pos.len() { for j in i+1..pos.len() {
        let d=model.displacement(pos[i],pos[j]); u+=model.pair(d[0].hypot(d[1]))?.0;
    }}
    let k=vel.iter().map(|v|0.5*(v[0]*v[0]+v[1]*v[1])).sum::<f64>();
    if !u.is_finite() || !k.is_finite() { return Err("nonfinite total energy".into()); }
    Ok((u,k))
}
