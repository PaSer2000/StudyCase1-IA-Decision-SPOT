# Resolution d'un problème de planification de prise de vue  sans incertitude
# Helene Fargier, oct 2025
#
#  Version bootstrap, avec exemple de declaration de modele, ajout de variables de decision, 
# d'une fonction objectif  ne prenant pas en compte les incertitudes 
# une seule contrainte implementée

# ================================================ #

import sys
import importlib

from pyscipopt import Model, quicksum
from itertools import product

# ================================================ #
# Chargement des données depuis un fichier choisi
# ================================================ #

if len(sys.argv) > 1:
    dataset_name = sys.argv[1].replace('.py', '') 
else:
    dataset_name = "spotProba1"

if len(sys.argv) > 2:
    critere_meteo = sys.argv[2].lower()
    if critere_meteo not in ['pessimiste', 'moyen', 'optimiste']:
        print(f"Attention : critère '{critere_meteo}' non reconnu.")
        print("Les choix valides sont : pessimiste, moyen, optimiste.")
        print("Utilisation de 'pessimiste' par défaut.")
        critere_meteo = "pessimiste"
else :
    critere_meteo = "pessimiste"

try:
    data = importlib.import_module(dataset_name)
    print(f"SUCCESS - Chargement des données depuis {dataset_name}")
except ImportError:
    print(f"ERREUR - Impossible de charger le fichier '{dataset_name}.py'")
    sys.exit(1)

nbImages = data.nbImages
nbInstruments = data.nbInstruments
PA = data.PA
DD = data.DD
AN = data.AN
VI = data.VI
DU = data.DU
TY = data.TY
PM = data.PM
PMmax = data.PMmax
Failure = data.Failure
ProbaInf = data.ProbaInf
ProbaSup = data.ProbaSup

# ================================================ #
# Creation du modèle lineaire
# ================================================ #

# model
mymodel = Model()

# pour chaque image i ,  le solveur doit affecter la variable booleen selection[i) à 1 ssi  l'image i est selectionnée
selection = {}
for i in range(nbImages):
    selection[i] = mymodel.addVar(vtype='B', name='select' + str(i))

# pour chaque image i ,  le solveur doit affecter la variable booleen selection[i) à 1 ssi  l'image i est selectionnée
assignedTo = {}
for i in range(nbImages):
    ass_i = {}
    for j in range(nbInstruments):
        ass_i[j] = mymodel.addVar(vtype='B', name='assignto' + str(i) + '_' + str(j))
    assignedTo[i] = ass_i

# ========================= #
# La fonction objectif
# ========================= #

# En l'absence d'incertitude, on maximise la somme des payoff
# La fonction objectif (avec prise en compte des incertitudes)

obj_expr = 0

for i in range(nbImages):
    if critere_meteo == 'pessimiste':
        p_nuage = ProbaSup[i]
    elif critere_meteo == 'optimiste':
        p_nuage = ProbaInf[i]
    elif critere_meteo == 'moyen':
        p_nuage = (ProbaInf[i] + ProbaSup[i]) / 2.0

    # IMAGE MONO : le gain dépend de l'instrument j réellement choisi
    if TY[i] == 1:
        for j in range(nbInstruments):
            esperance_gain = PA[i] * (1 - p_nuage) * (1 - Failure[j])
            obj_expr += esperance_gain * assignedTo[i][j]

    # IMAGE STEREO : dépend obligatoirement du succès des instruments 0 et 2   
    elif TY[i] == 2:
        esperance_gain = PA[i] * (1 - p_nuage) * (1 - Failure[0]) * (1 - Failure[2])
        # On l'attache directement à la variable de sélection
        obj_expr += esperance_gain * selection[i]

# On demande au solveur de maximiser cette espérance de gain
mymodel.setObjective(obj_expr, sense='maximize')

# ================================================ #
# Ajout des contraintes au modele
# ================================================ #

# la contrainte de non chevauchement
# considérons un instrument
# si, sur cet insrument, le temps de transition entre 2 images ima1 et ima2 
# ne tient pas entre la fin de ima1 et le debut de ima2 
# alors une seule de ces deux images au plus peut etre assignée à l'instrument

# C0 : Contrainte de non chevauchement
for ima1,ima2 in product(range(nbImages), range(nbImages)):
    if ima1 < ima2:
        for ins in range(nbInstruments):
            if  abs(DD[ima1][ins] - DD[ima2][ins]) * VI < DU * VI + abs(AN[ima1][ins] - AN[ima2][ins]):
                mymodel.addCons(assignedTo[ima1][ins] + assignedTo[ima2][ins] <= 1)
                

# C1 : Capacité mémoire
mymodel.addCons(
    quicksum(PM[i] * selection[i] for i in range(nbImages)) <= PMmax, 
    name="MemoireMax"
)

# C2 : Lien entre sélection et assignation + Contrainte des images stéréo
for i in range(nbImages):
    if TY[i] == 1:
        # IMAGE MONO : Si sélectionnée, elle doit être assignée à exactement 1 instrument
        mymodel.addCons(
            quicksum(assignedTo[i][j] for j in range(nbInstruments)) == selection[i],
            name=f"Mono_affectation_{i}"
        )
    elif TY[i] == 2:
        # IMAGE STEREO : Si sélectionnée, elle va sur l'avant (0) ET l'arrière (2)
        mymodel.addCons(assignedTo[i][0] == selection[i], name=f"Stereo_avant_{i}")
        mymodel.addCons(assignedTo[i][1] == 0, name=f"Stereo_nadir_interdit_{i}") # jamais sur nadir (1)
        mymodel.addCons(assignedTo[i][2] == selection[i], name=f"Stereo_arriere_{i}")
           
# ================================================ # 
# Résolution et affichage des resulats
# ================================================ #

#visualiser le problem lineaire cree
mymodel.writeProblem("pb.cip")

# lancer l'optimisation
print("Resolution")
mymodel.hideOutput(False)
mymodel.optimize()

#afficiher  les resultats mode "scip"
print('statut ' + mymodel.getStatus())
print("solution", end='\t')
print(mymodel.getBestSol())

# afficher les resultats prorement
if mymodel.getStatus() == 'optimal':
    print("\n\nProblème resolu, valeur de l'objectif " + str(mymodel.getObjVal()))
    sol=mymodel.getBestSol()
    for ima in range(nbImages):
        for ins in range(nbInstruments):
            if (mymodel.getVal(assignedTo[ima][ins]) > 0):
                print("Image" + str(ima) + " selectionnée et  assignée à  " + str(ins) + "  (debut à " + str( DD[ima][ins]) + ")")

