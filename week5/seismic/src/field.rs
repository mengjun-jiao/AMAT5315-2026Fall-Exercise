use ndarray::Array2;

#[derive(Debug, Clone)]
pub struct State {
    pub previous: Array2<f64>,
    pub current: Array2<f64>,
}

impl State {
    pub fn zeros(nz: usize, nx: usize) -> Self {
        let shape = (nz, nx);
        Self {
            previous: Array2::zeros(shape),
            current: Array2::zeros(shape),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::State;

    #[test]
    fn initial_state_is_zero_with_requested_shape() {
        let state = State::zeros(4, 5);

        assert_eq!(state.previous.shape(), &[4, 5]);
        assert_eq!(state.current.shape(), &[4, 5]);
        assert!(state.previous.iter().all(|value| *value == 0.0));
        assert!(state.current.iter().all(|value| *value == 0.0));
    }
}
