"""
Run the Fixed-Time model
On JiNan and HangZhou real data
"""

from utils.utils import traffic_r1_wrapper
import os
import time
from multiprocessing import Process
import argparse
from utils import error
import asyncio


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_name", type=str, default="TSCS-R1")
    parser.add_argument("--eightphase", action="store_true", default=False)
    # agent type, DIC_AGENTS keys in config.py
    parser.add_argument("--agent", type=str, default="LLMRule")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--gpt_version", type=str, default="gpt-4")
    parser.add_argument("--dataset", type=str, default="jinan")
    parser.add_argument("--traffic_file", type=str, default="anon_3_4_jinan_real.json")

    return parser.parse_args()


def main(in_args):
    traffic_file_list = []

    if in_args.dataset == "jinan":
        count = 3600
        road_net = "3_4"
        traffic_file_list = [
            "anon_3_4_jinan_real.json",  # jinan 1
            "anon_3_4_jinan_real_2000.json",  # jinan 2
            "anon_3_4_jinan_real_2500.json",  # jinan 3
            "anon_3_4_jinan_synthetic_24000_60min.json",
            "anon_3_4_jinan_synthetic_24h_6000.json",
        ]
        template = "Jinan"
    elif in_args.dataset == "hangzhou":
        count = 3600
        road_net = "4_4"
        traffic_file_list = [
            "anon_4_4_hangzhou_real.json",  # hangzhou 1
            "anon_4_4_hangzhou_real_5816.json",  # hangzhou 2
            "anon_4_4_hangzhou_synthetic_24000_60min.json",
        ]
        template = "Hangzhou"
    elif in_args.dataset == "newyork_28x7":
        count = 3600
        road_net = "28_7"
        traffic_file_list = [
            "anon_28_7_newyork_real_double.json",
            "anon_28_7_newyork_real_triple.json",
        ]
        template = "NewYork"

    # if in_args.prompt == "Commonsense":
    #     in_args.memo = "ChatGPTTLCSCommonsense"
    # elif in_args.prompt == "Wait Time Forecast":
    #     in_args.memo = "ChatGPTTLCWaitTimeForecast"
    in_args.memo = "TrafficR1"
    in_args.model = "TrafficR1"

    if "24h" in in_args.traffic_file:
        count = 86400

    # flow_file error
    try:
        if in_args.traffic_file not in traffic_file_list:
            raise error.flowFileException("Flow file does not exist.")
    except error.flowFileException as e:
        print(e)
        return

    NUM_ROW = int(road_net.split("_")[0])
    NUM_COL = int(road_net.split("_")[1])
    num_intersections = NUM_ROW * NUM_COL
    print("num_intersections:", num_intersections)
    print(in_args.traffic_file)

    log_dir = f"./r1_logs/{in_args.agent}/{in_args.gpt_version}/" + time.strftime(
        "%m%d_%H%M%S", time.localtime(time.time())
    )

    dic_agent_conf_extra = {
        "GPT_VERSION": in_args.gpt_version,
        "LOG_DIR": log_dir,
        "AGENT_TYPE": in_args.agent,
    }

    dic_traffic_env_conf_extra = {
        "NUM_AGENTS": num_intersections,
        "NUM_INTERSECTIONS": num_intersections,
        "MODEL_NAME": f"{in_args.model}-{dic_agent_conf_extra['GPT_VERSION']}",
        "PROJECT_NAME": in_args.proj_name,
        "RUN_COUNTS": count,
        "NUM_ROW": NUM_ROW,
        "NUM_COL": NUM_COL,
        "TRAFFIC_FILE": in_args.traffic_file,
        "ROADNET_FILE": "roadnet_{0}.json".format(road_net),
        "LIST_STATE_FEATURE": [
            "cur_phase",
            "traffic_movement_pressure_queue",
        ],
        "DIC_REWARD_INFO": {"pressure": 0},
    }

    if in_args.eightphase:
        dic_traffic_env_conf_extra["PHASE"] = {
            1: [0, 1, 0, 1, 0, 0, 0, 0],
            2: [0, 0, 0, 0, 0, 1, 0, 1],
            3: [1, 0, 1, 0, 0, 0, 0, 0],
            4: [0, 0, 0, 0, 1, 0, 1, 0],
            5: [1, 1, 0, 0, 0, 0, 0, 0],
            6: [0, 0, 1, 1, 0, 0, 0, 0],
            7: [0, 0, 0, 0, 0, 0, 1, 1],
            8: [0, 0, 0, 0, 1, 1, 0, 0],
        }
        dic_traffic_env_conf_extra["PHASE_LIST"] = [
            "WT_ET",
            "NT_ST",
            "WL_EL",
            "NL_SL",
            "WL_WT",
            "EL_ET",
            "SL_ST",
            "NL_NT",
        ]
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30, 30, 30, 30, 30]

    else:
        dic_agent_conf_extra["FIXED_TIME"] = [30, 30, 30, 30]

    dic_traffic_env_conf_extra["NUM_AGENTS"] = dic_traffic_env_conf_extra[
        "NUM_INTERSECTIONS"
    ]
    dic_path_extra = {
        "PATH_TO_MODEL": os.path.join(
            "model",
            in_args.memo,
            in_args.traffic_file
            + "_"
            + time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time())),
        ),
        "PATH_TO_WORK_DIRECTORY": os.path.join(
            "records",
            in_args.memo,
            in_args.traffic_file
            + "_"
            + time.strftime("%m_%d_%H_%M_%S", time.localtime(time.time())),
        ),
        "PATH_TO_DATA": os.path.join("data", template, str(road_net)),
    }

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    traffic_r1_wrapper(
        dic_agent_conf_extra,
        dic_traffic_env_conf_extra,
        dic_path_extra,
        f"{template}-{road_net}",
        in_args.traffic_file.split(".")[0],
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
    # asyncio.run(main(args))
