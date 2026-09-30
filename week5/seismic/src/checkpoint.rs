#[cfg(test)]
mod tests {
    use super::*;

    fn triples(schedule: &Schedule) -> Vec<(ActionKind, usize, usize)> {
        schedule
            .actions
            .iter()
            .map(|action| (action.action, action.step, action.saved_states))
            .collect()
    }

    #[test]
    fn binomial_capacity_and_tau_selection_match_reference() {
        assert_eq!(binomial_capacity(2, 2), 6);
        assert_eq!(binomial_capacity(3, 2), 10);
        assert_eq!(tau_for(6, 2).unwrap(), 2);
        assert_eq!(tau_for(240, 1).unwrap(), 239);
    }

    #[test]
    fn n6_delta2_matches_the_golden_action_sequence() {
        let schedule = generate_schedule(6, 2).unwrap();
        let expected = vec![
            (ActionKind::Call, 0, 1),
            (ActionKind::Call, 1, 1),
            (ActionKind::Call, 2, 1),
            (ActionKind::Store, 3, 2),
            (ActionKind::Call, 3, 2),
            (ActionKind::Call, 4, 2),
            (ActionKind::Store, 5, 3),
            (ActionKind::Grad, 5, 3),
            (ActionKind::Fetch, 5, 2),
            (ActionKind::Restore, 3, 2),
            (ActionKind::Call, 3, 2),
            (ActionKind::Store, 4, 3),
            (ActionKind::Grad, 4, 3),
            (ActionKind::Fetch, 4, 2),
            (ActionKind::Restore, 3, 2),
            (ActionKind::Grad, 3, 2),
            (ActionKind::Fetch, 3, 1),
            (ActionKind::Restore, 0, 1),
            (ActionKind::Call, 0, 1),
            (ActionKind::Store, 1, 2),
            (ActionKind::Call, 1, 2),
            (ActionKind::Store, 2, 3),
            (ActionKind::Grad, 2, 3),
            (ActionKind::Fetch, 2, 2),
            (ActionKind::Restore, 1, 2),
            (ActionKind::Grad, 1, 2),
            (ActionKind::Fetch, 1, 1),
            (ActionKind::Restore, 0, 1),
            (ActionKind::Grad, 0, 1),
        ];
        assert_eq!(schedule.tau, 2);
        assert_eq!(triples(&schedule), expected);
        let audit = audit_schedule(6, 2, &schedule.actions);
        assert!(audit.is_clean(), "{audit:?}");
        assert_eq!(audit.scheduler_forward_calls, 8);
        assert_eq!(audit.reverse_calls, 6);
        assert_eq!(audit.peak_saved_states, 3);
        assert_eq!(audit.final_saved, vec![0]);
    }

    #[test]
    fn edge_case_schedules_have_exact_reverse_order_and_final_s0() {
        for (steps, delta) in [(1, 1), (2, 1), (3, 1), (4, 2), (6, 2), (10, 3)] {
            let schedule = generate_schedule(steps, delta).unwrap();
            let audit = audit_schedule(steps, delta, &schedule.actions);
            assert!(audit.is_clean(), "N={steps}, delta={delta}: {audit:?}");
            assert_eq!(audit.grad_steps, (0..steps).rev().collect::<Vec<_>>());
            assert_eq!(audit.final_saved, vec![0]);
            assert!(audit.peak_saved_states <= delta + 1);
        }
        assert_eq!(generate_schedule(1, 1).unwrap().actions.len(), 1);
    }

    #[test]
    fn official_scheduler_budgets_and_marmousi_safety_pass_audit() {
        for delta in [1, 3, 5, 10] {
            let schedule = generate_schedule(240, delta).unwrap();
            let audit = audit_schedule(240, delta, &schedule.actions);
            assert!(audit.is_clean(), "delta={delta}: {audit:?}");
            assert_eq!(audit.reverse_calls, 240);
            assert_eq!(audit.peak_saved_states, delta + 1);
            println!(
                "N=240 delta={delta} tau={} actions={} scheduler_forward_calls={} reverse_calls={} peak_saved_states={}",
                schedule.tau,
                schedule.actions.len(),
                audit.scheduler_forward_calls,
                audit.reverse_calls,
                audit.peak_saved_states
            );
        }

        let schedule = generate_schedule(1200, 5).unwrap();
        let audit = audit_schedule(1200, 5, &schedule.actions);
        assert!(audit.is_clean(), "N=1200, delta=5: {audit:?}");
        assert_eq!(audit.reverse_calls, 1200);
        assert!(audit.peak_saved_states <= 6);
        println!(
            "N=1200 delta=5 tau={} actions={} scheduler_forward_calls={} peak_saved_states={}",
            schedule.tau,
            schedule.actions.len(),
            audit.scheduler_forward_calls,
            audit.peak_saved_states
        );
    }

    #[test]
    fn schedule_generation_is_deterministic_and_serializable() {
        let first = generate_schedule(10, 3).unwrap();
        let second = generate_schedule(10, 3).unwrap();
        assert_eq!(first, second);
        let json = serde_json::to_string(&first.actions).unwrap();
        assert!(json.contains(r#""action":"store""#));
        assert!(json.contains(r#""saved_states""#));
    }

    #[test]
    fn independent_auditor_reports_corrupt_actions() {
        let actions = vec![
            Action::new(ActionKind::Call, 0, 1),
            Action::new(ActionKind::Store, 1, 2),
            Action::new(ActionKind::Grad, 2, 1),
            Action::new(ActionKind::Restore, 4, 1),
            Action::new(ActionKind::Store, 0, 3),
            Action::new(ActionKind::Call, 3, 3),
            Action::new(ActionKind::Fetch, 0, 3),
        ];
        let audit = audit_schedule(3, 1, &actions);
        assert!(audit.grad_order_errors > 0);
        assert!(audit.invalid_restores > 0);
        assert!(audit.invalid_stores > 0);
        assert!(audit.invalid_calls > 0);
        assert!(audit.s0_fetches > 0);
        assert!(audit.saved_states_mismatches > 0);
        assert!(audit.budget_overruns > 0);
        assert!(audit.final_saved_set_errors > 0);
    }
}
use std::collections::BTreeSet;

use anyhow::{bail, Result};
use serde::Serialize;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum ActionKind {
    Store,
    Restore,
    Call,
    Grad,
    Fetch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct Action {
    pub action: ActionKind,
    pub step: usize,
    pub saved_states: usize,
}

impl Action {
    pub const fn new(action: ActionKind, step: usize, saved_states: usize) -> Self {
        Self {
            action,
            step,
            saved_states,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schedule {
    pub steps: usize,
    pub delta: usize,
    pub tau: usize,
    pub actions: Vec<Action>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditReport {
    pub grad_order_errors: usize,
    pub invalid_restores: usize,
    pub budget_overruns: usize,
    pub invalid_stores: usize,
    pub invalid_calls: usize,
    pub invalid_grads: usize,
    pub invalid_fetches: usize,
    pub s0_fetches: usize,
    pub saved_states_mismatches: usize,
    pub final_saved_set_errors: usize,
    pub scheduler_forward_calls: usize,
    pub reverse_calls: usize,
    pub peak_saved_states: usize,
    pub grad_steps: Vec<usize>,
    pub final_saved: Vec<usize>,
}

impl AuditReport {
    pub fn is_clean(&self) -> bool {
        self.grad_order_errors == 0
            && self.invalid_restores == 0
            && self.budget_overruns == 0
            && self.invalid_stores == 0
            && self.invalid_calls == 0
            && self.invalid_grads == 0
            && self.invalid_fetches == 0
            && self.s0_fetches == 0
            && self.saved_states_mismatches == 0
            && self.final_saved_set_errors == 0
    }
}

/// Return C(delta + tau, delta), saturating at usize::MAX.
pub fn binomial_capacity(delta: usize, tau: usize) -> usize {
    let total = match delta.checked_add(tau) {
        Some(value) => value,
        None => return usize::MAX,
    };
    let terms = delta.min(tau);
    let mut value = 1_u128;
    for index in 1..=terms {
        let factor = (total - terms + index) as u128;
        value = match value.checked_mul(factor) {
            Some(product) => product / index as u128,
            None => return usize::MAX,
        };
        if value > usize::MAX as u128 {
            return usize::MAX;
        }
    }
    value as usize
}

pub fn tau_for(steps: usize, delta: usize) -> Result<usize> {
    if steps == 0 {
        bail!("Treeverse requires a positive number of steps");
    }
    if delta == 0 && steps > 1 {
        bail!("Treeverse requires at least one additional checkpoint slot");
    }
    let mut tau = 1;
    while binomial_capacity(delta, tau) < steps {
        tau = tau
            .checked_add(1)
            .ok_or_else(|| anyhow::anyhow!("Treeverse tau overflow"))?;
    }
    Ok(tau)
}

fn ceil_div(numerator: usize, denominator: usize) -> Result<usize> {
    if denominator == 0 {
        bail!("Treeverse split denominator cannot be zero");
    }
    if numerator == 0 {
        Ok(0)
    } else {
        Ok((numerator - 1) / denominator + 1)
    }
}

fn split(delta: usize, tau: usize, sigma: usize, phi: usize) -> Result<usize> {
    let numerator = delta
        .checked_mul(sigma)
        .and_then(|value| value.checked_add(tau.checked_mul(phi)?))
        .ok_or_else(|| anyhow::anyhow!("Treeverse split numerator overflow"))?;
    let denominator = tau
        .checked_add(delta)
        .ok_or_else(|| anyhow::anyhow!("Treeverse split denominator overflow"))?;
    let mut kappa = ceil_div(numerator, denominator)?;
    if kappa >= phi && delta > 0 {
        kappa = (sigma + 1).max(phi.saturating_sub(1));
    }
    Ok(kappa)
}

struct Generator {
    steps: usize,
    saved: BTreeSet<usize>,
    working: usize,
    actions: Vec<Action>,
}

impl Generator {
    fn new(steps: usize) -> Self {
        let mut saved = BTreeSet::new();
        saved.insert(0);
        Self {
            steps,
            saved,
            working: 0,
            actions: Vec::new(),
        }
    }

    fn push(&mut self, action: ActionKind, step: usize) -> Result<()> {
        match action {
            ActionKind::Store => {
                if self.working != step || !self.saved.insert(step) {
                    bail!("invalid generated store at step {step}");
                }
            }
            ActionKind::Restore => {
                if !self.saved.contains(&step) {
                    bail!("invalid generated restore at step {step}");
                }
                self.working = step;
            }
            ActionKind::Call => {
                if self.working != step || step >= self.steps {
                    bail!("invalid generated call at step {step}");
                }
                self.working += 1;
            }
            ActionKind::Grad => {
                if !self.saved.contains(&step) {
                    bail!("invalid generated grad at step {step}");
                }
            }
            ActionKind::Fetch => {
                if step == 0 || !self.saved.remove(&step) {
                    bail!("invalid generated fetch at step {step}");
                }
            }
        }
        self.actions
            .push(Action::new(action, step, self.saved.len()));
        Ok(())
    }

    fn restore_if_needed(&mut self, step: usize) -> Result<()> {
        if self.working != step {
            self.push(ActionKind::Restore, step)?;
        }
        Ok(())
    }

    fn treeverse(
        &mut self,
        delta: usize,
        tau: usize,
        beta: usize,
        sigma: usize,
        mut phi: usize,
    ) -> Result<()> {
        let mut delta = delta;
        if sigma > beta {
            delta = delta
                .checked_sub(1)
                .ok_or_else(|| anyhow::anyhow!("Treeverse checkpoint depth underflow"))?;
            self.restore_if_needed(beta)?;
            for step in beta..sigma {
                self.push(ActionKind::Call, step)?;
            }
            self.push(ActionKind::Store, sigma)?;
        } else if sigma < beta {
            bail!("Treeverse split has sigma below beta");
        }

        let mut tau = tau;
        let mut kappa = split(delta, tau, sigma, phi)?;
        while tau > 0 && kappa < phi {
            self.treeverse(delta, tau, sigma, kappa, phi)?;
            tau -= 1;
            phi = kappa;
            kappa = split(delta, tau, sigma, phi)?;
        }

        self.restore_if_needed(sigma)?;
        self.push(ActionKind::Grad, sigma)?;
        if sigma > beta {
            self.push(ActionKind::Fetch, sigma)?;
        }
        Ok(())
    }
}

pub fn generate_schedule(steps: usize, delta: usize) -> Result<Schedule> {
    if steps == 0 {
        bail!("Treeverse requires a positive number of steps");
    }
    if delta == 0 {
        bail!("Treeverse requires at least one additional checkpoint slot");
    }
    let tau = tau_for(steps, delta)?;
    let mut generator = Generator::new(steps);
    generator.treeverse(delta, tau, 0, 0, steps)?;
    Ok(Schedule {
        steps,
        delta,
        tau,
        actions: generator.actions,
    })
}

pub fn audit_schedule(steps: usize, delta: usize, actions: &[Action]) -> AuditReport {
    let mut saved = BTreeSet::from([0]);
    let mut working = 0;
    let budget = delta.saturating_add(1);
    let mut report = AuditReport {
        grad_order_errors: 0,
        invalid_restores: 0,
        budget_overruns: 0,
        invalid_stores: 0,
        invalid_calls: 0,
        invalid_grads: 0,
        invalid_fetches: 0,
        s0_fetches: 0,
        saved_states_mismatches: 0,
        final_saved_set_errors: 0,
        scheduler_forward_calls: 0,
        reverse_calls: 0,
        peak_saved_states: 1,
        grad_steps: Vec::new(),
        final_saved: Vec::new(),
    };

    for action in actions {
        match action.action {
            ActionKind::Store => {
                if working != action.step || !saved.insert(action.step) {
                    report.invalid_stores += 1;
                }
            }
            ActionKind::Restore => {
                if saved.contains(&action.step) {
                    working = action.step;
                } else {
                    report.invalid_restores += 1;
                }
            }
            ActionKind::Call => {
                if working == action.step && action.step < steps {
                    working += 1;
                    report.scheduler_forward_calls += 1;
                } else {
                    report.invalid_calls += 1;
                }
            }
            ActionKind::Grad => {
                report.grad_steps.push(action.step);
                report.reverse_calls += 1;
                if action.step >= steps || !saved.contains(&action.step) {
                    report.invalid_grads += 1;
                }
            }
            ActionKind::Fetch => {
                if action.step == 0 {
                    report.s0_fetches += 1;
                    report.invalid_fetches += 1;
                } else if !saved.remove(&action.step) {
                    report.invalid_fetches += 1;
                }
            }
        }

        report.peak_saved_states = report.peak_saved_states.max(saved.len());
        if saved.len() > budget || action.saved_states > budget {
            report.budget_overruns += 1;
        }
        if action.saved_states != saved.len() {
            report.saved_states_mismatches += 1;
        }
    }

    let expected_grad_steps: Vec<_> = (0..steps).rev().collect();
    let compared = report.grad_steps.len().max(expected_grad_steps.len());
    report.grad_order_errors = (0..compared)
        .filter(|index| report.grad_steps.get(*index) != expected_grad_steps.get(*index))
        .count();
    report.final_saved = saved.iter().copied().collect();
    if report.final_saved != [0] {
        report.final_saved_set_errors = 1;
    }
    report
}
