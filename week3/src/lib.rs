use rand::Rng;

pub fn energy(spins: &[i8], l: usize) -> f64 {
    let mut e = 0i64;
    for row in 0..l {
        for col in 0..l {
            let s = i64::from(spins[row * l + col]);
            e -= s * i64::from(spins[row * l + (col + 1) % l]);
            e -= s * i64::from(spins[((row + 1) % l) * l + col]);
        }
    }
    e as f64
}

pub fn delta_energy(spins: &[i8], l: usize, row: usize, col: usize) -> i32 {
    let index = row * l + col;
    let s = i32::from(spins[index]);
    let neighbors = i32::from(spins[row * l + (col + l - 1) % l])
        + i32::from(spins[row * l + (col + 1) % l])
        + i32::from(spins[((row + l - 1) % l) * l + col])
        + i32::from(spins[((row + 1) % l) * l + col]);
    2 * s * neighbors
}

pub fn metropolis_sweep<R: Rng + ?Sized>(
    spins: &mut [i8],
    l: usize,
    temperature: f64,
    rng: &mut R,
) -> usize {
    let mut accepted = 0;
    for _ in 0..l * l {
        let index = rng.gen_range(0..l * l);
        let row = index / l;
        let col = index % l;
        let de = delta_energy(spins, l, row, col);
        if de <= 0 || rng.gen::<f64>() < (-(de as f64) / temperature).exp() {
            spins[index] = -spins[index];
            accepted += 1;
        }
    }
    accepted
}

pub fn magnetization(spins: &[i8]) -> f64 {
    spins.iter().map(|&s| f64::from(s)).sum::<f64>() / spins.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn periodic_neighbors_and_all_up_energy() {
        let spins = vec![1; 4];
        assert_eq!(delta_energy(&spins, 2, 0, 0), 8);
        assert_eq!(energy(&spins, 2), -8.0);
    }

    #[test]
    fn local_delta_matches_total_energy_difference() {
        let mut spins = vec![1, -1, 1, -1, -1, 1, -1, 1, 1];
        let before = energy(&spins, 3);
        let de = delta_energy(&spins, 3, 1, 1);
        spins[4] = -spins[4];
        assert_eq!(de as f64, energy(&spins, 3) - before);
    }

    #[test]
    fn seeded_sweep_repeats() {
        let mut a = vec![1; 16];
        let mut b = a.clone();
        let mut ra = ChaCha8Rng::seed_from_u64(2026);
        let mut rb = ChaCha8Rng::seed_from_u64(2026);
        assert_eq!(
            metropolis_sweep(&mut a, 4, 2.3, &mut ra),
            metropolis_sweep(&mut b, 4, 2.3, &mut rb)
        );
        assert_eq!(a, b);
    }
}
