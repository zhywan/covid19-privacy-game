# Privacy-Preserving COVID-19 Data Publishing Game

## Description:
This program finds the best solution for sharing individual-level COVID-19 case surveillance data, under an economically motivated adversary's re-identification attack based on a two-player Stackelberg game model named COVID-19 Re-Identification Game (CRIG). The attack is introduced by Sweeney in 2000 [1]. The adversary re-identifies the target with the help of a public demographic dataset (e.g., a voter registration list) by linking upon demographic attributes (e.g., year of birth and state of residence).

The game theoretic protection model is based upon Wan et al.'s Re-identification Game introduced in 2015 [2] and Wan et al.'s Multi-Stage Re-Identification Game (MSRIG) introduced in 2021 [3]. Two players in the game are a data subject and an adversary. The adversary's strategy is either to attack or not to attack, for each target. The data subject's strategy space is dependent upon specific scenarios in consideration.

## Usage:

The main program is "gamemodel_release.py". The programs for baselines are "noprotection_release.py", "dynamic_release.py", "cdc_release.py".

Note:

The cdc policy is obtained by using software ARX, a data anonymization tool (https://arx.deidentifier.org/).

The dynamic policy is obtained by using the "gamemodel_release.py" by setting the parameter COST to 0.

## References:

This code is partially based on our conference paper:

[0] A. Gourabathina, Z. Wan, J. T. Brown, C. Yan, and B. A. Malin. Privacy-Preserving Publishing of Individual-Level Pandemic Data Based on a Game Theoretic Model (under review).

Other published articles essential for understanding the software are as follows:

[1] L. Sweeney, Simple demographics often identify people uniquely. Health (San Francisco) 671(2000): 1-34, 2000.

[2] Z. Wan, Y. Vorobeychik, W. Xia, E. W. Clayton, M. Kantarcioglu, R. Ganta, R. Heatherly, and B. A. Malin. A game theoretic framework for analyzing re-identification risk. PloS one, 10(3): e0120592, 2015.

[3] Z. Wan, Y. Vorobeychik, W. Xia, Y. Liu, M. Wooders, J. Guo, Z. Yin, E. W. Clayton, M. Kantarcioglu, and B. A. Malin. Using game theory to thwart multistage privacy intrusions when sharing data. Sci. Adv. 7, eabe9986 (2021).
