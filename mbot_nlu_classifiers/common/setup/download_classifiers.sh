#!/bin/bash

# Change here when new classifiers are added
# Only .tar.gz file extensions are supported
# -> compress a directory using  tar -zcvf OUTPUT_FILE DIRECTORY
# If using nextcloud, need to add /download to the end of the url

CLASSIFIERS=(
    1 pedro_gpsr http://dantecloud.isr.tecnico.ulisboa.pt/index.php/s/3Y4pdcT9cm2AbY6/download
    2 mithun_gpsr http://dantecloud.isr.tecnico.ulisboa.pt/index.php/s/Zo9c2DzaFArTLKE/download
    3 mithun_gpsr_robocup http://dantecloud.isr.tecnico.ulisboa.pt/index.php/s/RR8zzjm6ddPXFFE/download
    4 mithun_eegpsr_robocup http://dantecloud.isr.tecnico.ulisboa.pt/index.php/s/4ioQP2S5knbzNy5/download
    5 mithun_erl_madrid http://dantecloud.isr.tecnico.ulisboa.pt/index.php/s/p2tiL7kBSZbE3Yk/download
    6 mithun_erl_l