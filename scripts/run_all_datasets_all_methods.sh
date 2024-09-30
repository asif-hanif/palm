#!/bin/bash
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/beijing_opera.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/crema_d.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/esc50_actions.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/esc50.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/gt_music_genre.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/ns_instruments.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/ravdess.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/sesa.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/tut.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/urban_sound.sh $MODEL;   done
for MODEL in zeroshot coop cocoop palm;  do  sh scripts/vocal_sound.sh $MODEL;   done