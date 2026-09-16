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

pub fn wolff_cluster_flip<R: Rng + ?Sized>(
    spins: &mut [i8],
    l: usize,
    temperature: f64,
    rng: &mut R,
) -> usize {
    let seed = rng.gen_range(0..l * l);
    let bond_probability = 1.0 - (-2.0 / temperature).exp();
    let mut in_cluster = vec![false; l * l];
    let mut stack = vec![seed];
    in_cluster[seed] = true;
    let original_spin = spins[seed];

    while let Some(index) = stack.pop() {
        let row = index / l;
        let col = index % l;
        let neighbors = [
            row * l + (col + l - 1) % l,
            row * l + (col + 1) % l,
            ((row + l - 1) % l) * l + col,
            ((row + 1) % l) * l + col,
        ];
        for neighbor in neighbors {
            if !in_cluster[neighbor]
                && spins[neighbor] == original_spin
                && rng.gen::<f64>() < bond_probability
            {
                in_cluster[neighbor] = true;
                stack.push(neighbor);
            }
        }
    }

    let mut cluster_size = 0;
    for (index, included) in in_cluster.into_iter().enumerate() {
        if included {
            spins[index] = -spins[index];
            cluster_size += 1;
        }
    }
    cluster_size
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

    #[test]
    fn wolff_flips_only_one_cluster_and_preserves_recomputed_observables() {
        let mut spins = vec![1, 1, -1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1];
        let before = spins.clone();
        let mut rng = ChaCha8Rng::seed_from_u64(17);
        let cluster_size = wolff_cluster_flip(&mut spins, 4, 2.3, &mut rng);
        let changed = before.iter().zip(&spins).filter(|(a, b)| a != b).count();
        assert!((1..=16).contains(&cluster_size));
        assert_eq!(changed, cluster_size);
        assert!(before.iter().zip(&spins).all(|(a, b)| a == b || *a == -*b));
        assert_eq!(
            magnetization(&spins),
            spins.iter().map(|&s| f64::from(s)).sum::<f64>() / 16.0
        );
        let direct_energy = (0..4)
            .flat_map(|row| (0..4).map(move |col| (row, col)))
            .map(|(row, col)| {
                let index = row * 4 + col;
                let neighbors = spins[row * 4 + (col + 3) % 4]
                    + spins[row * 4 + (col + 1) % 4]
                    + spins[((row + 3) % 4) * 4 + col]
                    + spins[((row + 1) % 4) * 4 + col];
                -f64::from(spins[index]) * f64::from(neighbors) / 2.0
            })
            .sum::<f64>();
        assert_eq!(energy(&spins, 4), direct_energy);
    }

    #[test]
    fn wolff_handles_periodic_l2_and_seeded_repetition() {
        let mut a = vec![1; 4];
        let mut b = a.clone();
        let mut ra = ChaCha8Rng::seed_from_u64(2026);
        let mut rb = ChaCha8Rng::seed_from_u64(2026);
        assert_eq!(wolff_cluster_flip(&mut a, 2, 0.01, &mut ra), 4);
        assert_eq!(wolff_cluster_flip(&mut b, 2, 0.01, &mut rb), 4);
        assert_eq!(a, b);
        assert_eq!(a, vec![-1; 4]);
    }
}
