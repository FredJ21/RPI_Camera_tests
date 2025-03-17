# -----------------------------------------------------------------------------
#
#
# -----------------------------------------------------------------------------
import enquiries
import re
import os
import subprocess

SRC_FILE    = "hailo_start.conf"

HAILO_ENV   = "hailo-rpi5-examples/setup_env.sh"
SCRIPTS_REP = "hailo-rpi5-examples/basic_pipelines/"

SWITCH_ADD  = "-f --arch hailo8"

# ----------------------------------------
def main():

    # -----------------------
    # Création de la liste
    #
    my_list = []
    f = open( SRC_FILE, 'r')
    for line in f:

        line = line.rstrip()
        if line and line[0] != "#":
            my_list.append(line)

    f.close

    # -----------------------
    # Affichage de la liste
    #
    print("\n\n")
    print("--------------------------------------------------")
    print("                 HAILO Tests")
    print("--------------------------------------------------")

    try: 
        my_script = enquiries.choose('', my_list)
    except:
        exit()


    # -----------------------
    # commande à excecuter
    #
    cmd = f"python {SCRIPTS_REP}{my_script}"
    cmd += f" {SWITCH_ADD}"

    print("\n\n")
    print("----------[ start script ]----------")
    print(cmd)
    print("------------------------------------")
    print("\n\n")

    # Ajoute l'environnement HAILO
    cmd  = f"source {HAILO_ENV};" + cmd

    
    # Attente <<enter>>
    tmp = input()


    subprocess.run(cmd, shell=True, executable="/bin/bash")


# ----------------------------------------
if __name__ == "__main__":
    main()




