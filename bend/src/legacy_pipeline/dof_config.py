"""DOF (Degrees of Freedom) configuration for motor unit control schemes."""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
import numpy as np


class DOFType(Enum):
    """Control scheme types for mapping motor units to DOFs."""
    SINGLE = "single"           # Single MU firing rate controls DOF
    MEAN = "mean"               # Mean FR of multiple MUs controls DOF (direction from mean)
    SUM = "sum"                 # Sum of FRs (both must be active to control)
    DIFFERENCE = "difference"   # Absolute difference between two MUs controls DOF
    COACTIVATION = "coactivation"  # Both MUs must be active (>threshold) to move, uses mean
    EXCLUSIVE = "exclusive"     # Only one MU active (>threshold) to move, uses that MU's FR


@dataclass
class DOFConfig:
    """Configuration for a single degree of freedom."""
    dof_type: DOFType
    motor_units: List[int]  # Indices of MUs used for this DOF (relative to selected MUs)
    name: str = ""          # Optional name for this DOF (e.g., "X-axis", "Y-axis")
    activation_threshold: float = 10.0  # Threshold for COACTIVATION mode (Hz)

    def compute_control_signal(self, firing_rates: np.ndarray, target_fr: float = 70.0) -> float:
        """
        Compute the control signal for this DOF based on firing rates.

        Args:
            firing_rates: Array of firing rates for all selected MUs
            target_fr: Target firing rate for normalization (max control at this FR)

        Returns:
            Normalized control signal in range [0, 1]
        """
        if not self.motor_units:
            return 0.0

        # Get firing rates for the MUs assigned to this DOF
        mu_frs = np.array([firing_rates[mu] for mu in self.motor_units])

        if self.dof_type == DOFType.SINGLE:
            # Single MU: direct mapping
            raw_signal = mu_frs[0]

        elif self.dof_type == DOFType.MEAN:
            # Mean of multiple MUs
            raw_signal = np.mean(mu_frs)

        elif self.dof_type == DOFType.SUM:
            # Sum of MUs (requires both to be active)
            raw_signal = np.sum(mu_frs)
            # Adjust target_fr for sum (expect higher combined FR)
            target_fr = target_fr * len(self.motor_units)

        elif self.dof_type == DOFType.DIFFERENCE:
            # Absolute difference between two MUs
            if len(mu_frs) >= 2:
                raw_signal = abs(mu_frs[0] - mu_frs[1])
            else:
                raw_signal = mu_frs[0]

        elif self.dof_type == DOFType.COACTIVATION:
            # Both MUs must be above threshold, then use mean
            if len(mu_frs) >= 2 and all(fr > self.activation_threshold for fr in mu_frs):
                raw_signal = np.mean(mu_frs)
            else:
                raw_signal = 0.0

        elif self.dof_type == DOFType.EXCLUSIVE:
            # Only one MU above threshold, use that MU's FR
            if len(mu_frs) >= 2:
                active_mask = mu_frs > self.activation_threshold
                n_active = np.sum(active_mask)
                if n_active == 1:
                    # Exactly one active - use its FR
                    raw_signal = mu_frs[active_mask][0]
                else:
                    # None or both active - no movement
                    raw_signal = 0.0
            else:
                raw_signal = mu_frs[0] if mu_frs[0] > self.activation_threshold else 0.0
        else:
            raw_signal = 0.0

        # Normalize to [0, 1]
        normalized = min(raw_signal / target_fr, 1.0)
        return normalized

    def describe(self) -> str:
        """Return a human-readable description of this DOF configuration."""
        mu_str = ", ".join([f"MU{mu}" for mu in self.motor_units])

        if self.dof_type == DOFType.SINGLE:
            desc = f"Single MU: {mu_str}"
        elif self.dof_type == DOFType.MEAN:
            desc = f"Mean of: {mu_str}"
        elif self.dof_type == DOFType.SUM:
            desc = f"Sum of: {mu_str}"
        elif self.dof_type == DOFType.DIFFERENCE:
            desc = f"Difference: |{mu_str}|"
        elif self.dof_type == DOFType.COACTIVATION:
            desc = f"Coactivation (>{self.activation_threshold:.0f}Hz): {mu_str}"
        elif self.dof_type == DOFType.EXCLUSIVE:
            desc = f"Exclusive (>{self.activation_threshold:.0f}Hz): {mu_str}"
        else:
            desc = f"Unknown: {mu_str}"

        if self.name:
            desc = f"{self.name}: {desc}"

        return desc


def prompt_dof_configuration(n_selected_mus: int, n_dofs: int = 2) -> List[DOFConfig]:
    """
    Interactively prompt the user to configure DOF control schemes.

    Args:
        n_selected_mus: Number of selected motor units available
        n_dofs: Number of DOFs to configure (default 2 for X/Y control)

    Returns:
        List of DOFConfig objects
    """
    print("\n" + "=" * 60)
    print("DOF CONFIGURATION")
    print("=" * 60)
    print(f"\nYou have {n_selected_mus} selected motor units (MU0 to MU{n_selected_mus - 1})")
    print(f"Configure {n_dofs} degrees of freedom for control.\n")

    print("Available control schemes:")
    print("  1. Single      - One MU's firing rate controls the DOF")
    print("  2. Mean        - Average FR of multiple MUs controls the DOF")
    print("  3. Sum         - Sum of FRs (both must be active for max control)")
    print("  4. Difference  - Absolute difference between two MUs controls the DOF")
    print("  5. Coactivation - Both MUs must be >threshold, then uses mean FR")
    print("  6. Exclusive   - Only one MU >threshold to move (not both)")
    print("-" * 60)

    dof_names = ["X-axis", "Y-axis"] if n_dofs == 2 else [f"DOF{i+1}" for i in range(n_dofs)]
    dof_configs = []

    for dof_idx in range(n_dofs):
        dof_name = dof_names[dof_idx]
        print(f"\n--- Configure {dof_name} ---")

        # Select control scheme type
        while True:
            try:
                scheme_input = input(f"Select scheme for {dof_name} (1-6): ").strip()
                scheme_num = int(scheme_input)
                if 1 <= scheme_num <= 6:
                    dof_type = [DOFType.SINGLE, DOFType.MEAN, DOFType.SUM, DOFType.DIFFERENCE, DOFType.COACTIVATION, DOFType.EXCLUSIVE][scheme_num - 1]
                    break
                else:
                    print("Please enter 1-6")
            except ValueError:
                print("Please enter a valid number")

        # Determine how many MUs are needed
        activation_threshold = 10.0  # Default threshold
        if dof_type == DOFType.SINGLE:
            n_mus_needed = 1
            print(f"Single scheme: Select 1 motor unit")
        elif dof_type == DOFType.MEAN:
            while True:
                try:
                    n_mus_needed = int(input(f"How many MUs to average? (2-{n_selected_mus}): ").strip())
                    if 2 <= n_mus_needed <= n_selected_mus:
                        break
                    print(f"Please enter a number between 2 and {n_selected_mus}")
                except ValueError:
                    print("Please enter a valid number")
        elif dof_type == DOFType.SUM:
            while True:
                try:
                    n_mus_needed = int(input(f"How many MUs to sum? (2-{n_selected_mus}): ").strip())
                    if 2 <= n_mus_needed <= n_selected_mus:
                        break
                    print(f"Please enter a number between 2 and {n_selected_mus}")
                except ValueError:
                    print("Please enter a valid number")
        elif dof_type == DOFType.DIFFERENCE:
            n_mus_needed = 2
            print(f"Difference scheme: Select 2 motor units")
        elif dof_type == DOFType.COACTIVATION:
            n_mus_needed = 2
            print(f"Coactivation scheme: Select 2 motor units")
            while True:
                try:
                    threshold_input = input(f"Activation threshold in Hz (default 10): ").strip()
                    if threshold_input == "":
                        activation_threshold = 10.0
                        break
                    activation_threshold = float(threshold_input)
                    if activation_threshold > 0:
                        break
                    print("Threshold must be positive")
                except ValueError:
                    print("Please enter a valid number")
        elif dof_type == DOFType.EXCLUSIVE:
            n_mus_needed = 2
            print(f"Exclusive scheme: Select 2 motor units (movement only when one is active)")
            while True:
                try:
                    threshold_input = input(f"Activation threshold in Hz (default 10): ").strip()
                    if threshold_input == "":
                        activation_threshold = 10.0
                        break
                    activation_threshold = float(threshold_input)
                    if activation_threshold > 0:
                        break
                    print("Threshold must be positive")
                except ValueError:
                    print("Please enter a valid number")

        # Select the motor units
        selected_mus = []
        for i in range(n_mus_needed):
            while True:
                try:
                    mu_input = input(f"  Select MU {i + 1} of {n_mus_needed} (0-{n_selected_mus - 1}): ").strip()
                    mu_idx = int(mu_input)
                    if 0 <= mu_idx < n_selected_mus:
                        if mu_idx in selected_mus:
                            print(f"  MU{mu_idx} already selected for this DOF, choose another")
                        else:
                            selected_mus.append(mu_idx)
                            break
                    else:
                        print(f"  Please enter a number between 0 and {n_selected_mus - 1}")
                except ValueError:
                    print("  Please enter a valid number")

        config = DOFConfig(
            dof_type=dof_type,
            motor_units=selected_mus,
            name=dof_name,
            activation_threshold=activation_threshold
        )
        dof_configs.append(config)
        print(f"  -> {config.describe()}")

    print("\n" + "-" * 60)
    print("DOF Configuration Summary:")
    for config in dof_configs:
        print(f"  {config.describe()}")
    print("-" * 60)

    return dof_configs


def serialize_dof_configs(configs: List[DOFConfig]) -> List[dict]:
    """Serialize DOF configs to a list of dicts for saving."""
    return [
        {
            "dof_type": config.dof_type.value,
            "motor_units": config.motor_units,
            "name": config.name,
            "activation_threshold": config.activation_threshold
        }
        for config in configs
    ]


def deserialize_dof_configs(data: List[dict]) -> List[DOFConfig]:
    """Deserialize DOF configs from a list of dicts."""
    return [
        DOFConfig(
            dof_type=DOFType(d["dof_type"]),
            motor_units=d["motor_units"],
            name=d.get("name", ""),
            activation_threshold=d.get("activation_threshold", 10.0)
        )
        for d in data
    ]


def compute_dof_control_signals(firing_rates: np.ndarray,
                                 dof_configs: List[DOFConfig],
                                 target_fr: float = 70.0) -> np.ndarray:
    """
    Compute control signals for all DOFs.

    Args:
        firing_rates: Array of firing rates for all selected MUs
        dof_configs: List of DOF configurations
        target_fr: Target firing rate for normalization

    Returns:
        Array of control signals, one per DOF
    """
    return np.array([
        config.compute_control_signal(firing_rates, target_fr)
        for config in dof_configs
    ])
