import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
import datetime
from typing import List, Tuple, Dict, Any
import pprint
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class ResourcePlanningEnv(gym.Env):
    metadata = {"render_modes": [], "render_fps": 4}

    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        self.config = config

        self.periods = config["Periods"]
        self.total_overall_budget_initial = float(config["Total_Overall_Budget"])
        self.limit_projects_per_period = config["Limit_Projects_Per_Period"]
        self.limit_sites_in_category_per_period = config["Limit_Sites_In_Region_Per_Period"]
        self.limit_total_sites_in_category = config["Limit_Total_Sites_In_Region"]

        self.category_names = list(config["Regions"].keys())
        self.n_categories = len(self.category_names)

        all_site_types_set = set()
        for cat_name in self.category_names:
            available_types_dict = config["Regions"][cat_name].get("Site_Types_Available", {})
            for site_type_name in available_types_dict.keys():
                all_site_types_set.add(site_type_name)
        self.global_site_type_names = sorted(list(all_site_types_set))

        self.possible_sites: List[Tuple[str, str]] = []
        self.site_details: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for cat_name in self.category_names:
            available_types_dict = config["Regions"][cat_name].get("Site_Types_Available", {})
            for site_type_name in available_types_dict:
                if site_type_name in self.global_site_type_names:
                    site_key = (cat_name, site_type_name)
                    self.possible_sites.append(site_key)
                    self.site_details[site_key] = available_types_dict[site_type_name]

        self.n_possible_sites = len(self.possible_sites)
        self.pass_action_index = self.n_possible_sites
        self.action_space = gym.spaces.Discrete(self.n_possible_sites + 1)

        obs_space_size = 1 + 1 + 1 + self.n_categories + self.n_categories + self.n_categories + self.n_possible_sites

        low_bounds = np.zeros(obs_space_size, dtype=np.float32)
        high_bounds = np.ones(obs_space_size, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low_bounds, high=high_bounds, dtype=np.float32)

        self.current_period = 0
        self.remaining_overall_budget = 0.0
        self.projects_built_this_period = 0
        self.sites_built_in_category_this_period = {}
        self.total_sites_built_in_category = {}
        self.initial_regional_budgets = {}
        self.remaining_regional_budgets = {}
        self.site_built_mask = np.zeros(self.n_possible_sites, dtype=np.int8)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_period = 0
        self.remaining_overall_budget = self.total_overall_budget_initial
        self.projects_built_this_period = 0

        self.initial_regional_budgets = {
            cat_name: float(cat_details.get("Initial_Regional_Budget", 0.0))
            for cat_name, cat_details in self.config["Regions"].items()
        }
        self.remaining_regional_budgets = self.initial_regional_budgets.copy()

        self.sites_built_in_category_this_period = {cat: 0 for cat in self.category_names}
        self.total_sites_built_in_category = {cat: 0 for cat in self.category_names}
        self.site_built_mask = np.zeros(self.n_possible_sites, dtype=np.int8)
        return self._get_obs(), self._get_info()

    def step(self, action: int):
        if not (0 <= action < self.action_space.n):
             print(f"ERROR: Invalid action {action}")
             return self._get_obs(), -100.0, True, False, {"error": "Invalid action index"}

        terminated = False
        truncated = False
        reward = 0.0
        site_built_successfully = False
        action_desc_for_info = "N/A"
        built_site_key_for_info = None

        if action == self.pass_action_index:
            action_desc_for_info = "PASS"
            self.current_period += 1
            self.projects_built_this_period = 0
            self.sites_built_in_category_this_period = {cat: 0 for cat in self.category_names}
            reward = 0.0
            if self.current_period >= self.periods:
                terminated = True
        else:
            site_idx = action
            if site_idx < 0 or site_idx >= self.n_possible_sites:
                 print(f"WARNING: Invalid site_idx {site_idx}")
                 return self._get_obs(), -10.0, False, False, {}

            category_name, site_type_name = self.possible_sites[site_idx]
            action_desc_for_info = f"BUILD {category_name}-{site_type_name}"
            details = self.site_details[(category_name, site_type_name)]

            overall_cost = float(details["Overall_Cost"])
            regional_cost_impact = float(details.get("Regional_Cost_Impact", 0.0))
            priority_score = float(details["Priority_Score"])

            can_build = True
            if self.site_built_mask[site_idx] == 1: can_build = False
            elif overall_cost > self.remaining_overall_budget: can_build = False
            elif regional_cost_impact > self.remaining_regional_budgets.get(category_name, 0): can_build = False
            elif self.projects_built_this_period >= self.limit_projects_per_period: can_build = False
            elif self.sites_built_in_category_this_period.get(category_name,0) >= self.limit_sites_in_category_per_period: can_build = False
            elif self.total_sites_built_in_category.get(category_name,0) >= self.limit_total_sites_in_category: can_build = False

            if can_build:
                self.remaining_overall_budget -= overall_cost
                self.remaining_regional_budgets[category_name] -= regional_cost_impact
                self.projects_built_this_period += 1
                self.sites_built_in_category_this_period[category_name] += 1
                self.total_sites_built_in_category[category_name] += 1
                self.site_built_mask[site_idx] = 1
                reward = priority_score
                site_built_successfully = True
                built_site_key_for_info = (category_name, site_type_name)
            else:
                reward = -1.0

        observation = self._get_obs()
        current_info = self._get_info()
        current_info['action_description'] = action_desc_for_info
        current_info['action_valid'] = site_built_successfully or (action == self.pass_action_index)
        current_info['site_built_key'] = built_site_key_for_info

        return observation, reward, terminated, truncated, current_info

    def _get_obs(self) -> np.ndarray:
        obs_list = []

        obs_list.append(self.current_period / max(1.0, float(self.periods - 1)) if self.periods > 0 else 0.0)

        obs_list.append(self.remaining_overall_budget / self.total_overall_budget_initial if self.total_overall_budget_initial > 0 else 0.0)

        obs_list.append(self.projects_built_this_period / float(self.limit_projects_per_period) if self.limit_projects_per_period > 0 else 0.0)

        for cat_name in self.category_names:
            obs_list.append(self.sites_built_in_category_this_period.get(cat_name, 0) / float(self.limit_sites_in_category_per_period) if self.limit_sites_in_category_per_period > 0 else 0.0)

        for cat_name in self.category_names:
            obs_list.append(self.total_sites_built_in_category.get(cat_name, 0) / float(self.limit_total_sites_in_category) if self.limit_total_sites_in_category > 0 else 0.0)

        for cat_name in self.category_names:
            initial_reg_budget = self.initial_regional_budgets.get(cat_name, 0.0)
            remaining_reg_budget = self.remaining_regional_budgets.get(cat_name, 0.0)
            if initial_reg_budget <= 0:
                norm_reg_budget = 1.0 if remaining_reg_budget > 1e-6 else 0.0
            else:
                norm_reg_budget = remaining_reg_budget / initial_reg_budget
            obs_list.append(np.clip(norm_reg_budget, 0.0, 1.0))

        obs_list.extend(self.site_built_mask.astype(np.float32))
        return np.array(obs_list, dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "current_period": self.current_period,
            "remaining_overall_budget": self.remaining_overall_budget,
            "remaining_regional_budgets": self.remaining_regional_budgets.copy(),
            "projects_built_this_period": self.projects_built_this_period,
            "sites_built_in_category_this_period": self.sites_built_in_category_this_period.copy(),
            "total_sites_built_in_category": self.total_sites_built_in_category.copy(),
            "site_built_mask_readable": {self.possible_sites[i]: int(self.site_built_mask[i]) for i in range(self.n_possible_sites) if i < len(self.possible_sites)},
            "pass_action_index": self.pass_action_index
        }
    def render(self): pass
    def close(self): pass
