//! Two-dimensional RDF averaged over a bounded window of saved frames.
use std::collections::VecDeque;
pub struct RdfWindow {
    n:usize,box_size:[f64;2],bins:usize,window:usize,
    history:VecDeque<Vec<usize>>,total:Vec<usize>,
}
pub struct RdfSample { pub centers:Vec<f64>,pub values:Vec<f64>,pub frames:usize }
impl RdfWindow {
    pub fn new(n:usize,box_size:[f64;2],bins:usize,window:usize)->Result<Self,String> {
        if n<2 || bins==0 || window==0 || box_size.iter().any(|l|!l.is_finite() || *l<=0.) { return Err("invalid RDF dimensions/window".into()); }
        Ok(Self{n,box_size,bins,window,history:VecDeque::new(),total:vec![0;bins]})
    }
    pub fn push(&mut self,pos:&[[f64;2]])->Result<RdfSample,String> {
        if pos.len()!=self.n || !pos.iter().flatten().all(|x|x.is_finite()) { return Err("invalid RDF positions".into()); }
        let max_r=self.box_size[0].min(self.box_size[1])/2.;let dr=max_r/self.bins as f64;
        let mut counts=vec![0;self.bins];
        let model=crate::physics::PhysicalModel::PeriodicShiftedLennardJones{box_size:self.box_size,rc:2.5};
        for i in 0..pos.len() { for j in i+1..pos.len() {
            let d=model.displacement(pos[i],pos[j]);let r=d[0].hypot(d[1]);
            if r==0. || !r.is_finite() { return Err("nonzero finite RDF separation required".into()); }
            if r<=max_r { counts[((r/dr).floor() as usize).min(self.bins-1)]+=1; }
        }}
        for (sum,&c) in self.total.iter_mut().zip(&counts) { *sum+=c; }
        self.history.push_back(counts);
        if self.history.len()>self.window {
            let old=self.history.pop_front().unwrap();
            for (sum,c) in self.total.iter_mut().zip(old) { *sum-=c; }
        }
        let rho=self.n as f64/(self.box_size[0]*self.box_size[1]);
        let mut centers=Vec::with_capacity(self.bins);let mut values=Vec::with_capacity(self.bins);
        for b in 0..self.bins {
            let inner=b as f64*dr;let outer=(b+1) as f64*dr;
            let area=std::f64::consts::PI*(outer*outer-inner*inner);
            centers.push((inner+outer)/2.);
            values.push(2.*self.total[b] as f64/(self.history.len() as f64*self.n as f64*rho*area));
        }
        Ok(RdfSample{centers,values,frames:self.history.len()})
    }
}
