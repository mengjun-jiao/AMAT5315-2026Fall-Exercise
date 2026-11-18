use serde_json::{json, Value};

fn spin(x: u8) -> i32 { 2 * i32::from(x) - 1 }
fn and(a:i32,b:i32,z:i32)->i32 { a*b-2*a*z-2*b*z-a-b+2*z }
fn or(a:i32,b:i32,z:i32)->i32 { a*b-2*a*z-2*b*z+a+b-2*z }
fn xor(a:i32,b:i32,z:i32,w:i32)->i32 { a*b-a*z-2*a*w-b*z-2*b*w+2*z*w-a-b+z+2*w }

pub fn enumerate()->Value {
    let mut rows=Vec::new();
    for name in ["AND","OR","NOT","COPY","SET0","SET1","XOR"] {
        let bits=match name {"AND"|"OR"|"XOR"=>3,"NOT"|"COPY"=>2,_=>1};
        for i in 0..(1<<bits) {
            let b:Vec<u8>=(0..bits).map(|j| ((i>>(bits-1-j))&1) as u8).collect();
            let s:Vec<i32>=b.iter().map(|&x|spin(x)).collect();
            let (energy, valid, helpers)=match name {
                "AND"=>(and(s[0],s[1],s[2]),b[2]==(b[0]&b[1]),vec![]),
                "OR"=>(or(s[0],s[1],s[2]),b[2]==(b[0]|b[1]),vec![]),
                "NOT"=>(s[0]*s[1],b[1]!=b[0],vec![]),
                "COPY"=>(-s[0]*s[1],b[1]==b[0],vec![]),
                "SET0"=>(s[0],b[0]==0,vec![]),
                "SET1"=>(-s[0],b[0]==1,vec![]),
                _=>{let e:Vec<i32>=[-1,1].iter().map(|&w|xor(s[0],s[1],s[2],w)).collect();
                    (e[0].min(e[1]),b[2]==(b[0]^b[1]),e)}
            };
            rows.push(json!({"gate":name,"bits":b,"energy":energy,"valid":valid,"helper_energies":helpers}));
        }
    }
    let mut circuit=Vec::new();
    for i in 0..16 {
        let b:Vec<u8>=(0..4).map(|j| ((i>>(3-j))&1) as u8).collect();
        let (a,bv,t,o)=(spin(b[0]),spin(b[1]),spin(b[2]),spin(b[3]));
        let energy=and(a,bv,t)+t*o-o;
        let valid=b[2]==(b[0]&b[1]) && b[3]!=b[2] && b[3]==1;
        circuit.push(json!({"bits":b,"energy":energy,"valid":valid}));
    }
    // Four inputs, six internal/output wires and two distinct XOR helpers.
    let mut cell=Vec::new();
    for i in 0..4096 {
        let b:Vec<u8>=(0..12).map(|j|((i>>(11-j))&1) as u8).collect();
        let s:Vec<i32>=b.iter().map(|&x|spin(x)).collect();
        let (p,q,a,bi,t,u,v,w,c,z,r,k)=(s[0],s[1],s[2],s[3],s[4],s[5],s[6],s[7],s[8],s[9],s[10],s[11]);
        let energy=and(p,q,t)+and(a,t,u)+xor(a,t,v,r)+and(bi,v,w)+or(u,w,c)+xor(v,bi,z,k);
        cell.push(json!({"bits":b,"energy":energy}));
    }
    json!({"gates":rows,"circuit":circuit,"multiplier_cell":{"order":["p","q","a","b","t","u","v","w","c","z","r","k"],"states":cell}})
}
