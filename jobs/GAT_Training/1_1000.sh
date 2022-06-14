#!/bin/sh
python ../../train_test_dgermen.py --id 309497 --bs 32 --dropout 0.0 --epoch 50 --factor 0.8 --fcl 128 --gcn_h 16 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 5 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 142043 --bs 16 --dropout 0.1 --epoch 100 --factor 0.5 --fcl 128 --gcn_h 128 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 417126 --bs 32 --dropout 0.1 --epoch 100 --factor 0.5 --fcl 128 --gcn_h 32 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 10 --weight_decay 0.001
python ../../train_test_dgermen.py --id 785350 --bs 64 --dropout 0.2 --epoch 100 --factor 0.5 --fcl 128 --gcn_h 16 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 20 --weight_decay 0.1
python ../../train_test_dgermen.py --id 268529 --bs 16 --dropout 0.2 --epoch 200 --factor 0.8 --fcl 512 --gcn_h 128 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 655775 --bs 64 --dropout 0.1 --epoch 30 --factor 0.8 --fcl 128 --gcn_h 64 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 10 --weight_decay 0.1
python ../../train_test_dgermen.py --id 752802 --bs 64 --dropout 0.2 --epoch 30 --factor 0.2 --fcl 64 --gcn_h 16 --lr 0.01 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 20 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 471494 --bs 32 --dropout 0.2 --epoch 30 --factor 0.8 --fcl 128 --gcn_h 64 --lr 0.01 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 20 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 14747 --bs 16 --dropout 0.0 --epoch 30 --factor 0.8 --fcl 512 --gcn_h 64 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 5 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 357735 --bs 32 --dropout 0.0 --epoch 200 --factor 0.8 --fcl 256 --gcn_h 32 --lr 0.01 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 5 --weight_decay 0.1
python ../../train_test_dgermen.py --id 304258 --bs 32 --dropout 0.0 --epoch 50 --factor 0.5 --fcl 256 --gcn_h 32 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 175107 --bs 16 --dropout 0.1 --epoch 200 --factor 0.8 --fcl 512 --gcn_h 16 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 782007 --bs 64 --dropout 0.2 --epoch 50 --factor 0.2 --fcl 512 --gcn_h 32 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 443108 --bs 32 --dropout 0.1 --epoch 200 --factor 0.5 --fcl 256 --gcn_h 128 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 10 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 50841 --bs 16 --dropout 0.0 --epoch 100 --factor 0.5 --fcl 256 --gcn_h 32 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.001
python ../../train_test_dgermen.py --id 724839 --bs 64 --dropout 0.1 --epoch 200 --factor 0.8 --fcl 128 --gcn_h 64 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 10 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 707490 --bs 64 --dropout 0.1 --epoch 100 --factor 0.2 --fcl 64 --gcn_h 32 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.1
python ../../train_test_dgermen.py --id 332787 --bs 32 --dropout 0.0 --epoch 100 --factor 0.8 --fcl 128 --gcn_h 32 --lr 0.01 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 88832 --bs 16 --dropout 0.0 --epoch 200 --factor 0.2 --fcl 256 --gcn_h 32 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 275068 --bs 16 --dropout 0.2 --epoch 200 --factor 0.2 --fcl 512 --gcn_h 32 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 475960 --bs 32 --dropout 0.2 --epoch 30 --factor 0.8 --fcl 512 --gcn_h 128 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 20 --weight_decay 0.1
python ../../train_test_dgermen.py --id 204021 --bs 16 --dropout 0.2 --epoch 30 --factor 0.2 --fcl 256 --gcn_h 32 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.001
python ../../train_test_dgermen.py --id 49163 --bs 16 --dropout 0.0 --epoch 100 --factor 0.5 --fcl 128 --gcn_h 64 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 314335 --bs 32 --dropout 0.0 --epoch 50 --factor 0.8 --fcl 512 --gcn_h 64 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.1
python ../../train_test_dgermen.py --id 213298 --bs 16 --dropout 0.2 --epoch 50 --factor 0.5 --fcl 512 --gcn_h 16 --lr 0.01 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 470764 --bs 32 --dropout 0.2 --epoch 30 --factor 0.8 --fcl 128 --gcn_h 16 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 686914 --bs 64 --dropout 0.1 --epoch 50 --factor 0.2 --fcl 128 --gcn_h 128 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 291690 --bs 32 --dropout 0.0 --epoch 30 --factor 0.8 --fcl 512 --gcn_h 128 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.1
python ../../train_test_dgermen.py --id 62384 --bs 16 --dropout 0.0 --epoch 100 --factor 0.2 --fcl 64 --gcn_h 32 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 20 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 759945 --bs 64 --dropout 0.2 --epoch 30 --factor 0.2 --fcl 512 --gcn_h 128 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 5 --weight_decay 0.1
python ../../train_test_dgermen.py --id 525692 --bs 32 --dropout 0.2 --epoch 100 --factor 0.2 --fcl 128 --gcn_h 128 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 336104 --bs 32 --dropout 0.0 --epoch 100 --factor 0.8 --fcl 512 --gcn_h 16 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 20 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 479771 --bs 32 --dropout 0.2 --epoch 30 --factor 0.2 --fcl 128 --gcn_h 128 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 20 --weight_decay 0.001
python ../../train_test_dgermen.py --id 322280 --bs 32 --dropout 0.0 --epoch 50 --factor 0.2 --fcl 512 --gcn_h 128 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.1
python ../../train_test_dgermen.py --id 441348 --bs 32 --dropout 0.1 --epoch 200 --factor 0.5 --fcl 128 --gcn_h 128 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 5 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 254676 --bs 16 --dropout 0.2 --epoch 200 --factor 0.5 --fcl 64 --gcn_h 64 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 10 --weight_decay 0.001
python ../../train_test_dgermen.py --id 539698 --bs 32 --dropout 0.2 --epoch 200 --factor 0.8 --fcl 128 --gcn_h 16 --lr 0.01 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 88651 --bs 16 --dropout 0.0 --epoch 200 --factor 0.2 --fcl 256 --gcn_h 16 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.001
python ../../train_test_dgermen.py --id 214172 --bs 16 --dropout 0.2 --epoch 50 --factor 0.5 --fcl 512 --gcn_h 64 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 754320 --bs 64 --dropout 0.2 --epoch 30 --factor 0.2 --fcl 64 --gcn_h 128 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.1
python ../../train_test_dgermen.py --id 650335 --bs 64 --dropout 0.1 --epoch 30 --factor 0.5 --fcl 256 --gcn_h 64 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.1
python ../../train_test_dgermen.py --id 142874 --bs 16 --dropout 0.1 --epoch 100 --factor 0.5 --fcl 256 --gcn_h 32 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 20 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 151489 --bs 16 --dropout 0.1 --epoch 100 --factor 0.8 --fcl 256 --gcn_h 128 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 5 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 402186 --bs 32 --dropout 0.1 --epoch 50 --factor 0.8 --fcl 128 --gcn_h 32 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 10 --weight_decay 0.001
python ../../train_test_dgermen.py --id 612202 --bs 64 --dropout 0.0 --epoch 100 --factor 0.8 --fcl 256 --gcn_h 128 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 273986 --bs 16 --dropout 0.2 --epoch 200 --factor 0.2 --fcl 256 --gcn_h 64 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.001
python ../../train_test_dgermen.py --id 325409 --bs 32 --dropout 0.0 --epoch 100 --factor 0.5 --fcl 128 --gcn_h 32 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 662660 --bs 64 --dropout 0.1 --epoch 30 --factor 0.2 --fcl 128 --gcn_h 16 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.1
python ../../train_test_dgermen.py --id 746499 --bs 64 --dropout 0.2 --epoch 30 --factor 0.8 --fcl 64 --gcn_h 128 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 10 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 153244 --bs 16 --dropout 0.1 --epoch 100 --factor 0.8 --fcl 512 --gcn_h 128 --lr 0.01 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 207930 --bs 16 --dropout 0.2 --epoch 50 --factor 0.5 --fcl 64 --gcn_h 32 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.1
python ../../train_test_dgermen.py --id 232371 --bs 16 --dropout 0.2 --epoch 100 --factor 0.5 --fcl 128 --gcn_h 16 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.001
python ../../train_test_dgermen.py --id 276323 --bs 16 --dropout 0.2 --epoch 200 --factor 0.2 --fcl 512 --gcn_h 128 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 765968 --bs 64 --dropout 0.2 --epoch 50 --factor 0.5 --fcl 256 --gcn_h 128 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 10 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 508377 --bs 32 --dropout 0.2 --epoch 100 --factor 0.5 --fcl 64 --gcn_h 128 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 754044 --bs 64 --dropout 0.2 --epoch 30 --factor 0.2 --fcl 64 --gcn_h 64 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 171061 --bs 16 --dropout 0.1 --epoch 200 --factor 0.8 --fcl 128 --gcn_h 16 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.001
python ../../train_test_dgermen.py --id 171244 --bs 16 --dropout 0.1 --epoch 200 --factor 0.8 --fcl 128 --gcn_h 16 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 531690 --bs 32 --dropout 0.2 --epoch 200 --factor 0.5 --fcl 64 --gcn_h 128 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.1
python ../../train_test_dgermen.py --id 67622 --bs 16 --dropout 0.0 --epoch 100 --factor 0.2 --fcl 512 --gcn_h 16 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 382611 --bs 32 --dropout 0.1 --epoch 30 --factor 0.8 --fcl 512 --gcn_h 32 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.001
python ../../train_test_dgermen.py --id 765301 --bs 64 --dropout 0.2 --epoch 50 --factor 0.5 --fcl 256 --gcn_h 64 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.001
python ../../train_test_dgermen.py --id 215286 --bs 16 --dropout 0.2 --epoch 50 --factor 0.8 --fcl 64 --gcn_h 16 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 10 --weight_decay 0.001
python ../../train_test_dgermen.py --id 731387 --bs 64 --dropout 0.1 --epoch 200 --factor 0.2 --fcl 64 --gcn_h 128 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 5 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 220414 --bs 16 --dropout 0.2 --epoch 50 --factor 0.8 --fcl 256 --gcn_h 128 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 821247 --bs 64 --dropout 0.2 --epoch 200 --factor 0.8 --fcl 512 --gcn_h 64 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 242089 --bs 16 --dropout 0.2 --epoch 100 --factor 0.8 --fcl 256 --gcn_h 16 --lr 0.01 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 5 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 404431 --bs 32 --dropout 0.1 --epoch 50 --factor 0.8 --fcl 256 --gcn_h 64 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.001
python ../../train_test_dgermen.py --id 302480 --bs 32 --dropout 0.0 --epoch 50 --factor 0.5 --fcl 128 --gcn_h 64 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.1
python ../../train_test_dgermen.py --id 556100 --bs 64 --dropout 0.0 --epoch 30 --factor 0.5 --fcl 128 --gcn_h 64 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.1
python ../../train_test_dgermen.py --id 301165 --bs 32 --dropout 0.0 --epoch 50 --factor 0.5 --fcl 64 --gcn_h 128 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.1
python ../../train_test_dgermen.py --id 571398 --bs 64 --dropout 0.0 --epoch 30 --factor 0.2 --fcl 128 --gcn_h 64 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 5 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 83876 --bs 16 --dropout 0.0 --epoch 200 --factor 0.8 --fcl 512 --gcn_h 64 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.001
python ../../train_test_dgermen.py --id 684236 --bs 64 --dropout 0.1 --epoch 50 --factor 0.2 --fcl 64 --gcn_h 32 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.001
python ../../train_test_dgermen.py --id 318797 --bs 32 --dropout 0.0 --epoch 50 --factor 0.2 --fcl 256 --gcn_h 16 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 5 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 13879 --bs 16 --dropout 0.0 --epoch 30 --factor 0.8 --fcl 512 --gcn_h 16 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 5 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 536431 --bs 32 --dropout 0.2 --epoch 200 --factor 0.5 --fcl 512 --gcn_h 32 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.001
python ../../train_test_dgermen.py --id 398146 --bs 32 --dropout 0.1 --epoch 50 --factor 0.5 --fcl 512 --gcn_h 32 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 5 --weight_decay 0.001
python ../../train_test_dgermen.py --id 817064 --bs 64 --dropout 0.2 --epoch 200 --factor 0.8 --fcl 128 --gcn_h 64 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 20 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 117957 --bs 16 --dropout 0.1 --epoch 50 --factor 0.5 --fcl 128 --gcn_h 32 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 603198 --bs 64 --dropout 0.0 --epoch 100 --factor 0.5 --fcl 256 --gcn_h 16 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 5 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 620151 --bs 64 --dropout 0.0 --epoch 100 --factor 0.2 --fcl 256 --gcn_h 128 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.001
python ../../train_test_dgermen.py --id 424334 --bs 32 --dropout 0.1 --epoch 100 --factor 0.8 --fcl 128 --gcn_h 16 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 20 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 658083 --bs 64 --dropout 0.1 --epoch 30 --factor 0.8 --fcl 256 --gcn_h 128 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 5 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 471978 --bs 32 --dropout 0.2 --epoch 30 --factor 0.8 --fcl 128 --gcn_h 128 --lr 0.01 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 5 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 152602 --bs 16 --dropout 0.1 --epoch 100 --factor 0.8 --fcl 512 --gcn_h 32 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 656279 --bs 64 --dropout 0.1 --epoch 30 --factor 0.8 --fcl 128 --gcn_h 128 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 20 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 576070 --bs 64 --dropout 0.0 --epoch 50 --factor 0.5 --fcl 64 --gcn_h 16 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 20 --weight_decay 0.1
python ../../train_test_dgermen.py --id 168502 --bs 16 --dropout 0.1 --epoch 200 --factor 0.5 --fcl 512 --gcn_h 128 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 10 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 211650 --bs 16 --dropout 0.2 --epoch 50 --factor 0.5 --fcl 256 --gcn_h 16 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 0.1
python ../../train_test_dgermen.py --id 646947 --bs 64 --dropout 0.1 --epoch 30 --factor 0.5 --fcl 64 --gcn_h 128 --lr 0.0001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 3 --patience 20 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 25445 --bs 16 --dropout 0.0 --epoch 50 --factor 0.5 --fcl 128 --gcn_h 32 --lr 0.1 --min_lr 2e-05 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 10 --weight_decay 0.1
python ../../train_test_dgermen.py --id 787183 --bs 64 --dropout 0.2 --epoch 100 --factor 0.5 --fcl 128 --gcn_h 128 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 20 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 440134 --bs 32 --dropout 0.1 --epoch 200 --factor 0.5 --fcl 128 --gcn_h 16 --lr 0.0001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 5 --weight_decay 1e-05
python ../../train_test_dgermen.py --id 665153 --bs 64 --dropout 0.1 --epoch 30 --factor 0.2 --fcl 256 --gcn_h 32 --lr 0.001 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 10 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 662153 --bs 64 --dropout 0.1 --epoch 30 --factor 0.2 --fcl 64 --gcn_h 128 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 3 --patience 10 --weight_decay 3e-06
python ../../train_test_dgermen.py --id 280537 --bs 32 --dropout 0.0 --epoch 30 --factor 0.5 --fcl 256 --gcn_h 16 --lr 0.01 --min_lr 0.0001 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 10 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 684547 --bs 64 --dropout 0.1 --epoch 50 --factor 0.2 --fcl 64 --gcn_h 64 --lr 0.1 --min_lr 0.0001 --model GATConv --num_of_ff_layers 1 --num_of_gcn_layers 2 --patience 10 --weight_decay 0.0001
python ../../train_test_dgermen.py --id 452917 --bs 32 --dropout 0.1 --epoch 200 --factor 0.8 --fcl 512 --gcn_h 128 --lr 0.001 --min_lr 2e-05 --model GATConv --num_of_ff_layers 2 --num_of_gcn_layers 2 --patience 10 --weight_decay 0.0001
