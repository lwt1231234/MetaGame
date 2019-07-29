\* rock_paper_scissors *\
Maximize
OBJ: vrs_w
Subject To
_C1: vrs_Seth >= 0
_C2: vrs_C._Viper >= 0
_C3: vrs_Cammy >= 0
_C4: vrs_C._Viper + vrs_Cammy + vrs_Seth = 1
_C5: 0.2 vrs_C._Viper + 0.4 vrs_Cammy - vrs_w >= 0
_C6: - 0.2 vrs_Cammy - 0.2 vrs_Seth - vrs_w >= 0
_C7: 0.2 vrs_C._Viper - 0.4 vrs_Seth - vrs_w >= 0
_C8: vrs_Cammy = 1
Bounds
vrs_C._Viper free
vrs_Cammy free
vrs_Seth free
vrs_w free
End
