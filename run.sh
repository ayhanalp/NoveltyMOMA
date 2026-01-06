#!/bin/bash

#conda activate env

#python main.py nsga2 rover \
#    "/home/santjami/repos/NoveltyMOMA/data/" \
#    "/home/santjami/repos/NoveltyMOMA/config/trap/4ag_trap_30kgens_DMOConfig.yaml" \
#    "/home/santjami/repos/NoveltyMOMA/config/trap/4ag_trap_30kgens_MORoverEnvConfig.yaml" \
#    2024 \
#    test1 \
#    10

python main.py nsga2 rover \
    "/home/santjami/repos/NoveltyMOMA/data/" \
    "/home/santjami/repos/NoveltyMOMA/config/trap/1ag_trap_30kgens_DMOConfig.yaml" \
    "/home/santjami/repos/NoveltyMOMA/config/trap/1ag_trap_30kgens_MORoverEnvConfig.yaml" \
    2024 \
    negh1ag_01beta_10k \
    50
