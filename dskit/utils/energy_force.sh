E0=$(grep '  without' OUTCAR | awk '{print $7}')
Ep=$(grep 'E_p' OSZICAR | tail -1 | awk '{print $3}')
N=$(sed -n '7p' POSCAR | awk '{for(i=1;i<=NF;i++)a+=$i; print a}')

echo "Generated by DeltaSpinKit (dskit)"
echo "RMS (uB):"
grep "RMS" OSZICAR | tail -1 | awk '{print $3}'
echo "Energy (eV):"
echo "$E0 $Ep" | awk '{printf"%.8f\n", ($1-$2)}'
echo "Cartesian Coordinates (Angstrom):"
grep "position of ions in cartesian coordinates" OUTCAR -A $N | tail -$N | awk '{print $1, $2, $3}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Weighted Magnetization (uB):"
grep 'MW_current' OSZICAR -A $N | tail -$N | awk '{print $2, $3, $4}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Weighted  Difference from Target (uB):"
grep 'MW_current' OSZICAR -A $N | tail -$N | awk '{print $5, $6, $7}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Magnetization (uB):"
grep 'M_current' OSZICAR -A $N | tail -$N | awk '{print $2, $3, $4}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Atomic Force (eV/A):"
grep 'TOTAL-FORCE' OUTCAR -A $(($N + 1)) | tail -$N | awk '{print $4, $5, $6}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Magnetic Force (eV/uB):"
grep 'Magnetic Force' OSZICAR -A $N | tail -$N | awk '{print $5, $6, $7}' | awk '{b[NR]=$0; }END{for(i=1;i<=NR;i++) if(i==NR){printf b[i] "\n"}else{printf "%s ",b[i]}}'
echo "Time Cost (sec):"
grep 'Elapsed time (sec)' OUTCAR | tail -1 | awk '{print $4}'
echo ""
