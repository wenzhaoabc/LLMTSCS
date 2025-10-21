import os
import copy
import requests
import json
import time
import re
import csv
import io
import pandas as pd
import numpy as np


from utils.my_utils import (
    load_json,
    dump_json,
    get_state_detail,
    get_state_three_segment,
)
from utils.cityflow_env import CityFlowEnv
from utils.llm import create_chat_completion

# url = "http://127.0.0.1:8000/v1/chat/completions"
url = os.getenv("LLM_API_URL", "http://127.0.0.1:8000/v1/chat/completions")
api_key = os.getenv("LLM_API_KEY", "sk--")
headers = {
    "Content-Type": "application/json",
    "Authorization": api_key,
}

four_phase_list = {"ETWT": 0, "NTST": 1, "ELWL": 2, "NLSL": 3}
eight_phase_list = {
    "ETWT": 0,
    "NTST": 1,
    "ELWL": 2,
    "NLSL": 3,
    "WTWL": 4,
    "ETEL": 5,
    "STSL": 6,
    "NTNL": 7,
}
location_dict = {"N": "North", "S": "South", "E": "East", "W": "West"}
location_dict_detail = {
    "N": "Northern",
    "S": "Southern",
    "E": "Eastern",
    "W": "Western",
}
direction_dict = {"T": "through", "L": "left-turn", "R": "turn-right"}
direction_dict_ori = {"T": "through", "L": "turn-left", "R": "turn-right"}

phase_explanation_dict_detail = {
    "NTST": "- NTST: Northern and southern through lanes.",
    "NLSL": "- NLSL: Northern and southern left-turn lanes.",
    "NTNL": "- NTNL: Northern through and left-turn lanes.",
    "STSL": "- STSL: Southern through and left-turn lanes.",
    "ETWT": "- ETWT: Eastern and western through lanes.",
    "ELWL": "- ELWL: Eastern and western left-turn lanes.",
    "ETEL": "- ETEL: Eastern through and left-turn lanes.",
    "WTWL": "- WTWL: Western through and left-turn lanes.",
}

incoming_lane_2_outgoing_road = {
    "NT": "South",
    "NL": "East",
    "ST": "North",
    "SL": "West",
    "ET": "West",
    "EL": "South",
    "WT": "East",
    "WL": "North",
}


class TrafficR1_Agent:
    def __init__(
        self, GPT_version, intersection, inter_name, phase_num, log_dir, dataset
    ):
        # init road length
        roads = copy.deepcopy(intersection["roads"])
        self.inter_name = inter_name
        self.roads = roads
        self.length_dict = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
        for r in roads:
            self.length_dict[roads[r]["location"]] = int(roads[r]["length"])

        self.phases = four_phase_list if phase_num == 4 else eight_phase_list

        self.gpt_version = GPT_version
        self.last_action = "ETWT"
        # self.system_prompt = load_json("./prompts/prompt_trafficr1.json")[
        #     "system_prompt"
        # ]
        self.prompt = load_json("./prompts/prompt_traffic_r1.json")

        self.state_action_prompt_file = f"{log_dir}/{dataset}-{self.inter_name}-{self.gpt_version}-{phase_num}_state_action_prompt_trafficr1.json"
        self.error_file = f"{log_dir}/{dataset}-{self.inter_name}-{self.gpt_version}-{phase_num}_error_prompts_trafficr1.json"
        self.state_action_prompt = []
        self.errors = []

    def choose_action(self, env: CityFlowEnv):
        state, state_incoming, avg_speed = get_state_detail(roads=self.roads, env=env)

        # 默认流量为0时使用ETWT
        flow_num = 0
        for road in state:
            flow_num += state[road]["queue_len"] + sum(state[road]["cells"])
        if flow_num == 0:
            action_code = self.action2code("ETWT")
            self.state_action_prompt.append(
                {"state": state, "prompt": [], "action": "ETWT"}
            )
            dump_json(self.state_action_prompt, self.state_action_prompt_file)
            return action_code

        signal_text = ""

        # chain-of-thought
        retry_counter = 0
        while signal_text not in self.phases:
            if retry_counter > 3:
                signal_text = "ETWT"
                break
            try:
                prompt = self.create_prompt(state)
                messages = [
                    {"role": "system", "content": self.prompt["system_prompt"]},
                    {"role": "user", "content": prompt},
                ]
                llm_res = create_chat_completion(
                    model=self.gpt_version,
                    messages=messages,
                    max_tokens=512 * 6,
                )
                llm_signal_text = llm_res.choices[0].message.content
                retry_counter += 1
                signal_answer_pattern = r"\\boxed{([^}]*)}"
                signal_text = re.findall(signal_answer_pattern, llm_signal_text)[-1]
                for s in self.phases.keys():
                    if s in signal_text.strip().upper():
                        signal_text = s
                        break

            except Exception as e:
                if "llm_res" not in locals():
                    llm_res = "No response"
                self.errors.append(
                    {
                        "error": str(e),
                        "prompt": prompt,
                        "response": llm_res.model_dump(),
                    }
                )
                dump_json(self.errors, self.error_file)
                # time.sleep(3)

        messages.append({"role": "assistant", "content": llm_res.model_dump()})
        action_code = self.action2code(signal_text)
        self.state_action_prompt.append(
            {
                "state": state,
                "state_incoming": state_incoming,
                "messages": messages,
                "action": signal_text,
            }
        )
        dump_json(self.state_action_prompt, self.state_action_prompt_file)

        self.temp_action_logger = action_code
        self.last_action = signal_text
        return action_code

    def action2code(self, action: str) -> int:
        code = self.phases[action]

        return code

    def create_prompt(self, state: dict) -> str:
        state_txt = f"""
## TSC Description
A crossroad connects two roads: the north-south and east-west. The traffic light is located at the intersection of the two roads. The north-south road is divided into two sections by the intersection: the north and south. Similarly, the east-west road is divided into the east and west. Each section has two lanes: a through and a left-turn lane. Each lane is further divided into three segments. Segment 1 is the closest to the intersection. Segment 2 is in the middle. Segment 3 is the farthest. In a lane, there may be early queued vehicles and approaching vehicles traveling in different segments. Early queued vehicles have arrived at the intersection and await passage permission. Approaching vehicles will arrive at the intersection in the future.

The traffic light has 4 signal phases. Each signal relieves vehicles' flow in the group of two specific lanes. The state of the intersection is listed below. It describes:
- The group of lanes relieving vehicles' flow under each traffic light phase.
- The number of early queued vehicles of the allowed lanes of each signal.
- The number of approaching vehicles in different segments of the allowed lanes of each signal.

## Traffic Observation
{self.state2table(state)}

## Task:
Which is the most effective traffic signal that will most significantly improve the traffic condition during the next phase, which relieves vehicles' flow of the allowed lanes of the signal?

## Note:
The traffic congestion is primarily dictated by the early queued vehicles, with the MOST significant impact. You MUST pay the MOST attention to lanes with long queue lengths. It is NOT URGENT to consider vehicles in distant segments since they are unlikely to reach the intersection soon.

## Format Instruction
You can only choose one of the signals listed above: NTST, NLSL, ETWT, ELWL. You FIRST think about the reasoning process for your choice as an internal monologue and then provide the final answer. Your think process MUST BE put in <think>...</think> tags. The final choice MUST BE put in \\boxed{{}}.
"""

        return state_txt

    def state2table(self, state: dict) -> str:
        state_txt = ""
        for p in self.phases:
            lane_1 = p[:2]
            lane_2 = p[2:]
            queue_len_1 = int(state[lane_1]["queue_len"])
            queue_len_2 = int(state[lane_2]["queue_len"])

            seg_1_lane_1 = state[lane_1]["cells"][0]
            seg_2_lane_1 = state[lane_1]["cells"][1]
            seg_3_lane_1 = state[lane_1]["cells"][2] + state[lane_1]["cells"][3]

            seg_1_lane_2 = state[lane_2]["cells"][0]
            seg_2_lane_2 = state[lane_2]["cells"][1]
            seg_3_lane_2 = state[lane_2]["cells"][2] + state[lane_2]["cells"][3]

            state_txt += (
                f"Signal: {p}\n"
                f"Allowed lanes: {phase_explanation_dict_detail[p][8:-1]}\n"
                f"- Early queued: {queue_len_1} ({location_dict[lane_1[0]]}), {queue_len_2} ({location_dict[lane_2[0]]}), {queue_len_1 + queue_len_2} (Total)\n"
                f"- Segment 1: {seg_1_lane_1} ({location_dict[lane_1[0]]}), {seg_1_lane_2} ({location_dict[lane_2[0]]}), {seg_1_lane_1 + seg_1_lane_2} (Total)\n"
                f"- Segment 2: {seg_2_lane_1} ({location_dict[lane_1[0]]}), {seg_2_lane_2} ({location_dict[lane_2[0]]}), {seg_2_lane_1 + seg_2_lane_2} (Total)\n"
                f"- Segment 3: {seg_3_lane_1} ({location_dict[lane_1[0]]}), {seg_3_lane_2} ({location_dict[lane_2[0]]}), {seg_3_lane_1 + seg_3_lane_2} (Total)\n\n"
            )

        return state_txt


class Rule_Agent:
    def __init__(
        self, GPT_version, intersection, inter_name, phase_num, log_dir, dataset
    ):
        # init road length
        roads = copy.deepcopy(intersection["roads"])
        self.inter_name = inter_name
        self.roads = roads
        self.length_dict = {"North": 0.0, "South": 0.0, "East": 0.0, "West": 0.0}
        for r in roads:
            self.length_dict[roads[r]["location"]] = int(roads[r]["length"])

        self.phases = four_phase_list if phase_num == 4 else eight_phase_list

        self.gpt_version = GPT_version
        self.last_action = "ETWT"

        self.state_action_prompt_file = f"{log_dir}/{dataset}-{self.inter_name}-rulebased-{phase_num}_state_action.json"
        self.error_file = (
            f"{log_dir}/{dataset}-{self.inter_name}-rulebased-{phase_num}_error.json"
        )
        self.state_action_prompt = []
        self.errors = []

    def choose_action(self, env: CityFlowEnv):
        state, state_incoming, avg_speed = get_state_detail(roads=self.roads, env=env)

        # 默认流量为0时使用ETWT
        flow_num = 0
        for road in state:
            flow_num += state[road]["queue_len"] + sum(state[road]["cells"])
        if flow_num == 0:
            action_code = self.action2code("ETWT")
            self.state_action_prompt.append(
                {"state": state, "action_reason": "Zero flow", "action": "ETWT"}
            )
            dump_json(self.state_action_prompt, self.state_action_prompt_file)
            return action_code

        # 计算各个相位的车流量
        phase_flow = {phase: 0 for phase in self.phases}
        for phase in phase_flow:
            phase_flow[phase] = (
                state[phase[:2]]["cells"][0] + state[phase[2:]]["cells"][0]
            )

        # 判断是否选择最大流量相位
        max_flow_phase = max(phase_flow, key=phase_flow.get)
        is_max_flow_phase = True
        for phase in phase_flow:
            if (
                phase != max_flow_phase
                and phase_flow[max_flow_phase] > phase_flow[phase] * 2.5
            ):
                is_max_flow_phase = False
                break
        if is_max_flow_phase:
            action_code = self.action2code(max_flow_phase)
            self.state_action_prompt.append(
                {
                    "state": state,
                    "action_reason": "Max flow phase",
                    "action": max_flow_phase,
                }
            )
            dump_json(self.state_action_prompt, self.state_action_prompt_file)
            return action_code

        # 判断是否选择最大排队长度相位
        phase_queue = {phase: 0.0 for phase in self.phases}
        phase_waiting_time = {phase: 0.0 for phase in self.phases}
        for phase in phase_queue:
            phase_queue[phase] = (
                state[phase[:2]]["queue_len"] + state[phase[2:]]["queue_len"]
            )
            phase_waiting_time[phase] = (
                state[phase[:2]]["avg_wait_time"] + state[phase[2:]]["avg_wait_time"]
            )

        sorted_queue_phase = sorted(
            phase_queue, key=lambda x: phase_queue[x], reverse=True
        )
        max_queue_phases = [
            f
            for f in phase_queue
            if phase_queue[f] == phase_queue[sorted_queue_phase[0]]
        ]
        if len(max_queue_phases) == 1:
            action_code = self.action2code(max_queue_phases[0])
            self.state_action_prompt.append(
                {
                    "state": state,
                    "action_reason": "Max queue phase",
                    "action": max_queue_phases[0],
                }
            )
            dump_json(self.state_action_prompt, self.state_action_prompt_file)
            return action_code

        # 选择等待时间最长的相位
        sorted_waiting_time_phase = sorted(
            phase_waiting_time,
            key=lambda x: phase_waiting_time[x],
            reverse=True,
        )
        max_waiting_time_phases = [
            f
            for f in phase_waiting_time
            if phase_waiting_time[f] == phase_waiting_time[sorted_waiting_time_phase[0]]
        ]
        # 默认选择第一个相位
        action_code = self.action2code(max_waiting_time_phases[0])
        self.state_action_prompt.append(
            {
                "state": state,
                "action_reason": "Max waiting time phase",
                "action": max_waiting_time_phases[0],
            }
        )
        dump_json(self.state_action_prompt, self.state_action_prompt_file)
        return action_code

    def action2code(self, action: str) -> int:
        code = self.phases[action]

        return code
