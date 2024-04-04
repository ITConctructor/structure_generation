#!/bin/bash
generate_dataset(){
    mkdir dataset
    cd dataset
    for ((i=0;i<$1;i++))
    do
    neper -T -format tesr -tesrformat ascii -id $i -n $2 -tesrsize $3
    done
}
declare -x -f generate_dataset