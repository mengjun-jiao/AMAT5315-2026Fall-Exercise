pub struct RdfWindow;
pub struct RdfSample { pub centers:Vec<f64>,pub values:Vec<f64>,pub frames:usize }
impl RdfWindow {
    pub fn new(_n:usize,_box_size:[f64;2],_bins:usize,_window:usize)->Result<Self,String> { unimplemented!("Part 4 RDF") }
    pub fn push(&mut self,_pos:&[[f64;2]])->Result<RdfSample,String> { unimplemented!("Part 4 RDF window") }
}
