C 'MAGMA.FOR'
C written by Al Cameron in 1987
C modified by Bruce Fegley July 2002
C bfegley@wustl.edu
C modified by Laura Schaefer August 2002, June 2003(save file)
C laura_s@levee.wustl.edu
C thermodynamic data comments added by Bruce Fegley June 2004
C go to end of file for thermodynamic & physical property data
C modified by Laura Schaefer March 2005 - added Fe2O3 and Fe3O4,
C added commenting, and set CON(El) to zero when A(El) is zero
C References describing the MAGMA code:
C B. Fegley, Jr. & A.G.W. Cameron (1987) A Vaporization Model for 
C Iron/Silicate Fractionation in the Mercury Protoplanet. 
C Earth Planet. Sci. Lett. 82, 207-222.
C L. Schaefer & B. Fegley, Jr. (2004)A Thermodynamic Model of High
C Temperature Lava Vaporization on Io. Icarus 169, 216-241.


	IMPLICIT DOUBLE PRECISION(A-H, M, O-Z)
	REAL AMDATA(25),AMELTD(44)
	CHARACTER TYPE*25
	CHARACTER SOURCE*50
	CHARACTER NAME*10
C	PCONV converts the pressures into number densities
C	PCONV is (dyn/cm**2=>atm) / Boltzmann's constant (R/AVOG)
C	AVOG is Avogadro's number
	PCONV = 1.01325D6 / 1.38046D-16
	AVOG = 6.023D23
	IREP = 2
C ISTEP counts the number of vaporization steps
	ISTEP = 0
C IPRN is used to determine which steps are printed to the output file
	IPRN = 0
C IIT counts the number of iterations needed to solve for the activities
	IIT = 0
	IFIRST = 0
	
	
	OPEN(1,FILE = 'MAGMA.OUT')
	OPEN(2,FILE = '(V:1600)MAGMA.DAT',FORM = 'UNFORMATTED')
	OPEN(3,FILE = '(V:1600)MELT.DAT',FORM = 'UNFORMATTED')

	
C	weights per metal atom of the magma molecules (mol wt. in g/mole)
	WMGSIO3 = 1.0040D2 / AVOG
	WMG2SIO4 = 0.5D0 * 1.4071D2 / AVOG
	WMGO = 4.031D1 / AVOG
	WFESIO3 = 1.3193D2 / AVOG
	WFE2SIO4 = 0.5D0 * 2.0378D2 / AVOG
	WFEO = 7.185D1 / AVOG
	WFE = 5.5847D1 / AVOG
	WCAO = 5.608D1 / AVOG
	WAL2O3 = 0.5D0 * 1.0196D2 / AVOG
	WTIO2 = 7.990D1 / AVOG
	WNA2O = 6.198D1 / AVOG
	WK2O = 9.420D0 / AVOG

C	Molecular weights of the oxides
	MWSiO2 = 60.084
	MWAl2O3 = 101.961
	MWTiO2 = 79.865
	MWFe2O3 = 159.689
	MWFeO = 71.845
	MWMgO = 40.304
	MWCaO = 56.077
	MWNa2O = 61.979
	MWK2O = 94.195
	
C150	CONTINUE

C  Prompt for temperature - all calculations are isothermal
	WRITE(*,10)
10	FORMAT(1X,'INPUT T  ')
	READ(*,20)T
	
C  input temperature in Kelvin, up to 3 decimal places, e.g., 1652.234
C  and write to the ouput file

20	FORMAT(F10.3)
	WRITE(1,300)T
300	FORMAT(1X,'T = ',1PE13.6)


C  Ask whether composition should be imported from a file (1) or 
C  entered manually (0)
	WRITE(*,100)
100	FORMAT(1X,'Take composition from file? (N=0,Y=1)')
	READ(*,*)TAKE
	
	IF (TAKE.EQ.0)THEN
	GOTO 101
	ELSE
	GOTO 102
	ENDIF
	
C  If using a composition from a file, enter the file name (6 letters all caps + .CMP)
102	WRITE(*,*)'File name: '
	READ(*,FMT='(A)')NAME
	OPEN(4,FILE = NAME)
	READ(4,106)TYPE,SOURCE,WTSiO2,WTMgO,WTAl2O3,WTTiO2,
     *  WTFe2O3,WTFeO,WTCaO,WTNa2O,WTK2O
        CLOSE(4)
        GOTO 103

C  Input the weight% of the oxides manually, up to 5 decimal places, e.g., 22.99999
C  add comments for the type of magma and literature source for the composition
101	WRITE(*,*) 'MAGMA TYPE'
	READ(*,FMT='(A)') TYPE
	WRITE(*,*) 'SOURCE:'
	READ(*,FMT='(A)') SOURCE
	WRITE(*,*) 'INPUT OXIDE WEIGHT %'
	WRITE(*,*) 'SiO2'
	READ(*,25) WTSiO2
	WRITE(*,*) 'MgO'
	READ(*,25) WTMgO
	WRITE(*,*) 'Al2O3'
	READ(*,25) WTAl2O3
	WRITE(*,*) 'TiO2'
	READ(*,25) WTTiO2
	WRITE(*,*) 'Fe2O3'
	READ(*,25) WTFe2O3
	WRITE(*,*) 'FeO'
	READ(*,25) WTFeO
	WRITE(*,*) 'CaO'
	READ(*,25) WTCaO
	WRITE(*,*) 'Na2O'
	READ(*,25) WTNa2O
	WRITE(*,*) 'K2O'
	READ(*,25) WTK2O
     
C  Option to save the current magma composition to a file
C  should be used if the current composition will be used more than once
	WRITE(*,*)' Save composition to file? (N=0,Y=1)'
	READ(*,*)SAVE
	IF(SAVE.EQ.0)THEN
	GOTO 103
	ELSE
	GOTO 104
	ENDIF
C  Assign a file name for the composition. Name should be no more than 
C  6 letters (all caps) plus the extension .CMP
104	WRITE(*,*)'File name?(6 letters + .CMP)'
	READ(*,FMT='(A)')NAME
	OPEN(4, FILE = NAME)
	WRITE(4,106)TYPE,SOURCE,WTSiO2,WTMgO,WTAl2O3,WTTiO2,
     *  WTFe2O3,WTFeO,WTCaO,WTNa2O,WTK2O
        CLOSE(4)
        
106	FORMAT(A25,/,A50,/,F8.5,/,F8.5,/,F8.5,/,F8.5,/,F8.5,/,F8.5,/,
     *  F8.5,/,F8.5,/,F8.5)        
25	FORMAT(F8.5)

C  Number of moles of the oxides calculated from the input composition
103	TOTWT=WTSiO2+WTMgO+WTAl2O3+WTTiO2+WTFe2O3+WTFeO+WTCaO+WTNa2O+
     *  WTK2O
	MOSiO2 = WTSiO2 / MWSiO2
	MOMgO = WTMgO / MWMgO
	MOAl2O3 = WTAl2O3 / MWAl2O3
	MOTiO2 = WTTiO2 / MWTiO2
	MOFe2O3 = WTFe2O3 / MWFe2O3
	MOFeO = WTFeO / MWFeO
	MOCaO = WTCaO / MWCaO
	MONa2O = WTNa2O / MWNa2O
	MOK2O = WTK2O / MWK2O
	TotMol = MOSiO2 + MOMgO + MOAl2O3 + MOTiO2 + MOFe2O3 + MOFeO
     * + MOCaO + MONa2O + MOK2O
	
C	Mole % of the oxides
	MPSiO2 = 100 * MOSiO2 / TotMol
	MPMgO = 100 * MOMgO / TotMol
	MPAl2O3 = 100 * MOAl2O3 / TotMol
	MPTiO2 = 100 * MOTiO2 / TotMol
	MPFe2O3 = 100 * MOFe2O3 / TotMol
	MPFeO = 100 * MOFeO / TotMol
	MPCaO = 100 * MOCaO / TotMol
	MPNa2O = 100 * MONa2O / TotMol
	MPK2O = 100 * MOK2O / TotMol
	TOTAL=MPSiO2+MPMgO+MPAl2O3+MPTiO2+MPFe2O3+MPFeO+MPCaO+MPNa2O+
     *  MPK2O
	
C	Elemental Abundances normalized to Si = 1E6
C	AE(element) = (# of metal atoms in oxide) * MO(moles of oxide)
C			* 1D6 / moles of SiO2
C 	If there is no Si in the composition, do not normalize to Si.

	IF (MOSiO2 .NE. 0.0D0) THEN
	AESI = 1D6
	AEMG = MOMgO * 1D6 / MOSiO2
	AEAL = MOAl2O3 * 2 * 1D6 / MOSiO2
	AETI = MOTiO2 * 1D6 / MOSiO2
	AEFE = (MOFeO + 2 * MOFe2O3) * 1D6 / MOSiO2
	AECA = MOCaO * 1D6 / MOSiO2
	AENA = 2 * MONa2O * 1D6 / MOSiO2
	AEK = 2 * MOK2O * 1D6 / MOSiO2
	ELSE IF (MOSiO2 .EQ. 0.0D0) THEN
	AESI = MOSiO2 * AVOG
	AEMG = MOMgO * AVOG
	AEAL = MOAl2O3 * 2 *AVOG
	AETI = MOTiO2 * AVOG
	AEFE = (MOFeO + 2 * MOFe2O3)* AVOG
	AECA = MOCaO * AVOG
	AENA = 2 * MONa2O * AVOG
	AEK = 2 * MOK2O * AVOG
	ENDIF	
		
C	renormalize the abundances
C	AETOT = total atomic abundance of all of the elements (except O)
C	AETOT1 = molecular abundace of all of the oxides

	AETOT = AESI + AEMG + AEFE + AECA + AEAL + AETI + 
     * AENA + AEK
	AETOT1 = AESI + AEMG + AEFE + AECA + 0.5D0*AEAL 
     * + AETI + 0.5D0*AENA + 0.5D0*AEK

C	calculate initial weight of melt so that wetight% vaporized
C	can be calculated later
	MASS0 = AESI * MWSiO2 + AEMG * MWMgO + AEFE * MWFeO +  
     * AECA * MWCaO + AEAL * 0.5 * MWAl2O3  
     * + AETI * MWTiO2 + 0.5 * AENA * MWNa2O + 0.5 * AEK * MWK2O

C	PLANETARY ABUNDANCES = ELEMENTAL ABUNDANCES
C       (This is an artifact from original code)
	PLANSI = AESI
	PLANMG = AEMG
	PLANFE = AEFE
	PLANCA = AECA
	PLANAL = AEAL
	PLANTI = AETI
	PLANNA = AENA
	PLANK = AEK
	PLMANT = AETOT
	PLMAN0 = PLMANT
	PLANRAT = PLMANT / PLMAN0

C	Write the initial composition and source data to output file

	WRITE (1,FMT=15)
15	FORMAT(1X,'MAGMA COMPOSITION:')
	WRITE(1,FMT='(A)') TYPE
	WRITE(1, FMT='(A)') 'SOURCE:', SOURCE
	WRITE(1,35)
35	FORMAT(/,1X, 'OXIDE','      WT%','             MOLE%')
	WRITE(1,45) WTSiO2,MPSiO2,WTMgO,MPMgO,WTAl2O3,MPAl2O3,WTTiO2,
     *	MPTiO2,WTFe2O3,MPFe2O3,WTFeO,MPFeO,WTCaO,MPCaO,WTNa2O,MPNa2O,
     *  WTK2O,MPK2O,TOTWT,TOTAL 
45	FORMAT(1X, 'SiO2   ',1PE13.6,4X,1PE13.6,/,' MgO    ',1PE13.6,4X
     *  ,1PE13.6,/,' Al2O3  ',1PE13.6,4X,1PE13.6,/,' TiO2   ',1PE13.6,
     *  4X,1PE13.6,/,' Fe2O3  ',1PE13.6,4X,1PE13.6,/,' FeO    ',1PE13.6
     *  ,4X,1PE13.6,/,' CaO    ',1PE13.6,4X,1PE13.6,/,' Na2O   ',
     *  1PE13.6,4X,1PE13.5,/,' K2O    ',1PE13.6,4X,1PE13.6,/,
     *  ' TOTAL  ',1PE13.6,4X,1PE13.6)
	WRITE(1,320)
	WRITE(1,330)PLANSI,PLANMG,PLANFE,PLANCA,PLANAL,
     * PLANTI,PLANNA,PLANK
	WRITE(*,330)PLANSI,PLANMG,PLANFE,PLANCA,PLANAL,
     * PLANTI,PLANNA,PLANK
	
C	relative abundances of the metals (by molecule) in the mantle
C	same as oxide mole fractions in the magma

	FSI = AESI / AETOT1
	FMG = AEMG / AETOT1
	FFE = AEFE / AETOT1
	FCA = AECA / AETOT1
	FAL = 0.5D0 * AEAL / AETOT1
	FTI = AETI / AETOT1
	FNA = 0.5D0 * AENA / AETOT1
	FK = 0.5D0 * AEK / AETOT1
	
C	relative abundances of the metals (by atom)
	CONSI = AESI / AETOT
	CONMG = AEMG / AETOT
	CONFE = AEFE / AETOT
	CONCA = AECA / AETOT
	CONAL = AEAL / AETOT
	CONTI = AETI / AETOT
	CONNA = AENA / AETOT
	CONK = AEK / AETOT
	
C      	write the mole fractions of metal oxides in the mantle

	WRITE(1,360)
360	FORMAT('Oxide Mole Fraction (F) in Silicate',/)	

	WRITE(1,400)FSI,FMG,FFE,FCA,FAL,FTI,FNA,FK
	WRITE(*,400)FSI,FMG,FFE,FCA,FAL,FTI,FNA,FK
400	FORMAT('FSiO2 = ',1PE13.6,/,'FMgO =  ',1PE13.6,/,'FFeO =  ',
     * 1PE13.6,/,'FCaO =   ',1PE12.6,/,'FAl2O3 = ',1PE12.6,/,
     * 'FTiO2 =  ',1PE12.6,/,'FNa2O = ',1PE13.6,/,'FK2O =  ',
     * 1PE13.6,/)


C	RELATIVE ATOMIC ABUNDANCES OF METALS

	WRITE(1,409)
	WRITE(1,410)CONSI,CONMG,CONFE,CONCA,CONAL,CONTI,CONNA,CONK
    	WRITE(*,410)CONSI,CONMG,CONFE,CONCA,CONAL,CONTI,CONNA,CONK
         
409	FORMAT(/,'RELATIVE ATOMIC ABUNDANCES OF METALS',/)

410	FORMAT(1X,' CONSI = ',1PD13.4,'  CONMG = ',1PD13.4,'  CONFE = ',
     * 1PD13.4,/,'  CONCA = ',1PD13.4,'  CONAL = ',1PD13.4,'  CONTI = ',
     * 1PD13.4,/,'  CONNA = ',1PD13.4,'  CONK  = ',1PD13.4,/)

	AMDATA(1) = PLANRAT
	AMDATA(2) = CONSI
	AMDATA(3) = CONMG
	AMDATA(4) = CONFE
	AMDATA(5) = CONCA
	AMDATA(6) = CONAL
	AMDATA(7) = CONTI
	AMDATA(16) = CONNA
	AMDATA(17) = CONK
	
C	assume initial values of key pressures
	PSIOG = 1.0D0
	PO2G = 1.0D0
	PMGOG = 1.0D0
	PFEG = 1.0D0
	PCAG = 1.0D0
	PALG = 1.0D0
	PTIG = 1.0D0
	PNAG = 1.0D0
	PKG = 1.0D0
	
C	initial unit values of adjustment factors
	ASIOG = 1.0D0
	AO2G = 1.0D0
	AMGOG = 1.0D0
	AFEG = 1.0D0
	ACAG = 1.0D0
	AALG = 1.0D0
	ATIG = 1.0D0
	ANAG = 1.0D0
	AKG = 1.0D0
	
C	initial values of activity coefficients, gamma
	GAMSI = 1.0D0
	GAMMG = 1.0D0
	GAMFE = 1.0D0
	GAMFE3 = 1.0D0
	GAMCA = 1.0D0
	GAMAL = 1.0D0
	GAMTI = 1.0D0
	GAMNA = 1.0D0
	GAMK = 1.0D0
	
	GMSI = 1.0D0
	GMMG = 1.0D0
	GMFE = 1.0D0
	GMFE3 = 1.0D0
	GMCA = 1.0D0
	GMAL = 1.0D0
	GMTI = 1.0D0
	GMNA = 1.0D0
	GMK = 1.0D0
	
	
C  COMP is a counting factor. When COMP is 1, assume all Fe-oxide is FeO,
C  and compute the activities for all metal oxides. Then compute the 
C  activities of Fe2O3 and Fe3O4 with the gas chemistry. 
C  These activities are then put back into main activity calculations, 
C  and the activity of FeO (and all other oxides) is adjusted. 

	COMP = 0.0D0
	
	
	
C  GAS CHEMISTRY THERMODYNAMIC DATA

	
C	begin silicon and oxygen
C  SiO2(liq) = Si(g) + 2O(g)
	A = 22.13 - 94311.0 / T
C JANAF 2nd ed. & supplements 2000-4500 K every 500 degrees 
C correlation coefficient for linear fit = -0.99989
C	AK1 = 10.0**A = P(SIG)*P(OG)**2/P(SIO2L)

C  0.5O2 (g) = O(g)
	E = 3.47 - 13282 / T
C JANAF 2nd ed. 
C correlation coefficient for linear fit = 
C	AK2 = 10.0**E = P(OG)/P(02G)**0.5
	EOG = 10.0**E
C	HENCE 10.0**A = P(SIG)*(P(O2G)*10.0**2E)/P(SIO2L)
C	HENCE P(SIO2L) = 10.0**(2.0*E-A)*P(SIG)*P(O2G)
C	HENCE P(O2G) = 10.0**(-2.0*E) * P(OG)**2

C  Si(liq) = Si(g)
	B = 6.00 - 20919 / T
C	AK3 = 10.0**B = P(SIG)/P(SIL)
C	HENCE P(SIG) = 10.0**B*P(SIL)
	ESIG = 10.0**B
C  Si(liq) + 0.5 O2(g) = SiO(g)
	C = 2.51 + 8207 / T
C	FOR P(SIG) INSTEAD OF P(SIL)	C2 = - 3.67 + 29760 / T
C	AK4 = 10.0**C = P(SIOG)/(P(SIL)*P(O2G)**0.5)
C	HENCE P(SIL) = 10.0**-C*P(SIOG)/P(O2G)**0.5
	ESIL = 10.0**(-C)
C	HENCE P(SIO2L) = 10.0**(2.0*E+B-A)*10.0**(-C)*P(SIOG)*P(O2G)**0.5
C	HENCE P(SIO2L) = 10.0**(2.0*E+B-A-C)*P(SIOG)*P(O2G)**0.5
	ESIO2L = 10.0**(2.0D0*E + B - A - C)
C  Si(liq) + O2(g) = SiO2(g)
	D = -1.44 + 18326 / T
C	FOR P(SIG) INSTEAD OF P(SIL)	D2 = - 7.57 + 39595 / T
C	AK5 = 10.0**D = P(SIO2G)/(P(SIL)*P(O2G))
C	HENCE P(SIO2G) = 10.0**D*P(SIL)*P(O2G)
	ESIO2G = 10.0**D
C	HENCE P(SIO2G) = 10.0**D*10.0**-B*P(SIG)*P(O2G)
C	HENCE P(SIO2G) = 10.0**(D-B) * P(SIG) * P(O2G)

C	begin magnesium

C MgO(liq) = Mg(g) + O(g)
	F = 12.56 - 46992 / T
C JANAF 2nd ed. & supplements 1500-5000 K every 500 degrees 
C correlation coefficient for linear fit = -0.99994
C	AK6 = 10.0**F = P(MGG)*P(OG)/P(MGOL)
C	HENCE P(MGOL) = 10.0**(-F) * P(MGG) * P(OG)
	EMGOL = 10.0**(-F)
	
C  Mg(g) + 0.5 O2(g) = MgO(g)
	G = -1.19 + 3794 / T
C	AK8 = 10.0**G = P(MGOG)/(P(MGG)*P(O2G)**0.5)
C	HENCE P(MGOG) = 10.0**G * P(MGG) * P(O2G)**0.5
	EMGG = 10.0**(-G)

C	begin iron

C  FeO(liq) = Fe(g) + O(g)
	AA = 12.06 - 44992 / T
C JANAF 2nd ed. & supplements 2000-5000 K every 500 degrees 
C correlation coefficient for linear fit = -0.99995
C	AK9 = 10.0**AA = P(FEG)*P(OG)/P(FEOL)
C	HENCE P(FEOL) = 10.0**(-AA) * P(FEG) * P(OG)
	EFEOL = 10.0**(-AA)
	
C  Fe(liq) = Fe(g)
	AB = 6.35 -19704 / T
C	AK10 = 10.0**AB = P(FEG)/P(FEL)
	EFEL = 10.0**(-AB)
C  Fe(liq) + 0.5 O2(g) = FeO(g)
	AC = 3.39 - 9951 / T
C	FOR P(FEG) INSTEAD OF P(FEL)	AC2 = - 2.93 + 9945 / T
C	AK11 = 10.0**AC = P(FEOG)/(P(FEL)*P(O2G)**0.5)
C	HENCE P(FEOG) = 10.0**(AC-AB) * P(FEG) * P(O2G)**0.5
	EFEOG = 10.0**(AC-AB)

C 2Fe (g) + 1.5 O2 (g) = Fe2O3 (liq)
C Fe2O3 (liq) cp data from IVTANTHERMO database (estimated)
C hematite enthalpy of fusion calculated from Sugawara & Akaogi 2004
	AD = -2.26722053113D01 + 7.56430936141329D04 / T
C       K = PFE2O3L / (PFEG**2 * PO2G**0.5)
C	HENCE P(FE2O3L) = 10**AD * PFEG**2 * PO2G**0.5
	EFE2O3L = 10**AD
	
C 3Fe (g) + 2 O2 (g) = Fe3O4 (liq)
C Fe3O4 (liq) cp data from Barin 95
C magnetite enthalpy of fusion from JANAF 4th ed.
	AE = -3.19907301154D01 + 1.110526206139634D05 / T
C	K = PFE3O4L / (PFEG**3 * PO2G**0.5)
C	HENCE P(FE3O4L) = 10**AE * PFEG**3 * PO2G**0.5
	EFE3O4L = 10**AE

C	begin calcium

C  CaO(liq) = Ca(g) + O(g)
	BA = 11.88 - 49586 / T
C JANAF 2nd ed. & supplements 2000-4500 K every 500 degrees 
C correlation coefficient for linear fit = -0.99998
C	AK12 = 10.0**BA = P(CAG)*P(OG)/P(CAOL)
C	HENCE P(CAOL) = 10.0**(-BA) * P(CAG) * P(OG)
	ECAOL = 10.0**(-BA)
	
C  Ca(g) + 0.5 O2(g) = CaO(g)
	BC = -1.61 + 6128 / T
C	AK14 = 10.0**BC = P(CAOG)/(P(CAG)*P(O2G)**0.5)
C	HENCE P(CAOG) = 10.0**BC * P(CAG) * P(O2G)**0.5
	ECAOG = 10.0**BC

C	begin aluminum

C  Al2O3(liq) = 2Al(g) + 3O(g)
	CA = 35.83 - 153255 / T
C JANAF supplements 1500-4000 K every 500 degrees 
C correlation coefficient for linear fit = -0.99995
C	AK15 = 10.0**CA = P(ALG)**2*P(OG)**3/P(AL2O3L)
C	HENCE P(AL2O3L) = 10.0**(-CA) * P(ALG)**2 * P(OG)**3
	EAL2O3L = 10.0**(-CA)
	
C  Al(liq) = Al(g)
	CB = 5.70 - 15862 / T
C	AK16 = 10.0**CB = P(ALG)/P(ALL)
	EALL = 10.0**(-CB)
C  Al(liq) + 0.5 O2(g) = AlO(g)
	CC = 3.04 - 2143 / T
C	FOR P(ALG) INSTEAD OF P(ALL)	CC2 = - 2.43 + 13067 / T
C	AK17 = 10.0**CC = P(ALOG)/(P(ALL)*P(O2G)**0.5)
C	HENCE P(ALOG) = 10.0**(CC-CB) * P(ALG) * P(O2G)**0.5
	EALOG = 10.0**(CC-CB)
C  Al(liq) + O2(g) = AlO2(g)
	CD = - 0.09 + 5.523 / T
C	FOR P(ALG) INSTEAD OF P(ALL)	CD2 = - 5.70 + 21159 / T
C	AK18 = 10.0**CD = P(ALO2G)/(P(ALL)*P(O2G))
C	HENCE P(ALO2G) = 10.0**(CD-CB) * P(ALG) * P(O2G)
	EALO2G = 10.0**(CD-CB)
C  2Al(liq) + 0.5 O2(g) = Al2O(g)
	CE = 2.04 + 10232 / T
C	FOR P(ALG) INSTEAD OF P(ALL)	CE2 = - 9.32 + 41897 / T
C	AK19 = 10.0**CE = P(AL2OG)/P(ALL)**2 * P(O2G)**0.5
C	HENCE P(AL2OG) = 10.0**(CE-2.0D0*CB) * P(ALG)**2 * P(O2G)**0.5
	EAL2OG = 10.0**(CE - 2.0D0 * CB)
C  2Al(liq) + O2(g) = Al2O2(g)
	CF = - 1.53 + 23021 / T
C	FOR P(ALG) INSTEAD OF P(ALL)	CF2 = - 12.86 + 54600 / T
C	AK20 = 10.0**CF = P(AL2O2G)/(P(ALL)**2 * P(O2G))
C	HENCE P(AL2O2G) = 10.0**(CF-2.0D0*CB) * P(ALG)**2 * PO2G
	EAL2O2G = 10.0**(CF - 2.0D0 * CB)

C	begin titanium

C  Ti(liq) + 0.5 O2(g) = TiO(g)
	DC = 4.31 - 2101 / T
C	FOR P(TIG) INSTEAD OF P(TIL)	DC2 = - 3.33 + 23747 / T
C	AK22 = 10.0**DC = P(TIOG)/(P(TIL)*P(O2G)**0.5)
C	HENCE P(TIL) = 10.0**(-DC) * P(TIOG) / P(O2G)**0.5
	ETIOG = 10.0**DC
C  Ti(liq) = Ti(g)
	DB = 6.46 - 23025 / T
C	AK21 = 10.0**DB = P(TIG)/P(TIL)
	ETIL = 10.0**(-DB)
	
C  TiO2(liq) = Ti(g) + 2O(g)
	DA = 21.07 - 95362 / T
C JANAF 2nd ed. & supplements 1500-4000 K every 500 degrees 
C correlation coefficient for linear fit = -0.99998
C	AK20 = 10.0**DA = P(TIG)*P(OG)**2/P(TIO2L)
C	HENCE P(TIO2L) = 10.0**(-DA) * P(TIG) * P(OG)**2
	ETIO2L = 10.0**(-DA)
	
C  Ti(liq) + O2(g) = TiO2(g)
	DD = - 0.41 + 17926 / T
C	FOR P(TIG) INSTEAD OF P(TIL)	DD2 = - 7.44 + 43028 / T
C	AK23 = 10.0**DD = P(TIO2G)/(P(TIL)*P(O2G))
C	HENCE P(TIO2G) = 10.0**DD * P(TIL) * P(O2G)
	ETIO2G = 10.0**DD

C	Begin sodium
C  2Na(g) + O(g) = Na2O(liq)
	EA = - 15.56 + 40286/T
C JANAF 2nd ed. & supplements
C correlation coefficient for linear fit = 
	ENA2OL = 10.0**EA
C  Na(g) + O(g) = NaO(g)
	EB = - 1.43 + 1287/T
	ENAOG = 10.0**(EB-E)
C  2Na(g) = Na2(g)
	EC = - 4.31 + 4281/T
	ENA2G = 10.0**EC
C  2Na(g) + O(g) = Na2O(g)
	ED = - 7.00 + 11898/T
	ENA2OG = 10.0**(ED-E)
C  Na(g) = Na+(g) + e-(g)
	EE = 2.80 - 27851/T
	ENACAT = 10**EE
	
C	Begin potassium
C  2K(g) + O(g) = K2O(liq)
	FA = - 15.33 + 36735/T
	EK2OL = 10.0**FA
C  K(g) + O(g) = KO(g)
	FB = - 1.28 + 959/T
	EKOG = 10.0**(FB-E)
C  2K(g) = K2(g)
	FC = - 3.94 + 2852/T
	EK2G = 10.0**FC
C  2K(g) + O(g) = K2O(g)
	FD = - 10.734 + 30817/T
	EK2OG = 10.0**(FD-E)
C  K(g) = K+(g) + e-(g)
	FE = 2.76 - 23760/T
	EKCAT = 10**FE	

C	Begin thorium
C  Th(liq) + O2(g) = ThO2(liq)
	GA = - 9.55 + 63948/T
	ETHO2L = 10.0**GA
C  Th(g) = Th(liq)
	GB = - 5.96 + 29600/T
	ETHL = 10.0**GB
C  Th(liq) + 0.5 O2(g) = ThO(g)
	GC = 2.75 + 3497/T
	ETHOG = 10.0**GC
C  Th(liq) + O2(g) = ThO2(g)
	GD = - 1.58 + 28875/T
	ETHO2G = 10.0**GD

C	Begin uranium
C  U(liq) + O2(g) = UO2(liq)
	HA = - 26.91 + 204359/T
	EUO2L = 10.0**HA
C  U(g) = U(liq)
	HB = - 5.75 + 25470/T
	EUL = 10.0**HB
C  U(liq) + 0.5 O2(g) = UO(g)
	HC = 3.02 + 1705/T
	EUOG = 10.0**HC
C  U(liq) + O2(g) = UO2(g)
	HD = - 1.19 + 26554/T
	EUO2G = 10.0**HD
C  U(liq) + 1.5 O2(g) = UO3(g)
	HE = - 4.24 +43710/T
	EUO3G = 10.0**HE

C	Begin plutonium
C  Pu(liq) + O2(g) = PuO2(liq)
	QA = - 29.86 + 200903/T
	EPUO2L = 10.0**QA
C  Pu(g) = Pu(liq)
	QB = - 4.79 + 17316/T
	EPUL = 10.0**QB
C  Pu(liq) + 0.5 O2(g) = PuO(g)
	QC = 2.40 + 6875/T
	EPUOG = 10.0**QC
C  Pu(liq) + O2(g) = PuO2(g)
	QD = - 1.76 + 25984/T
	EPUO2G = 10.0**QD



C  THERMODYNAMIC DATA FOR ACTIVITIES IN THE MELT

C  Prototype:
C  2MgO(liq) + SiO2(liq) = Mg2SiO4(liq)
C  log10K(Mg2SiO4) = - 34.08 + 141582/T
C  -2log10K(MgO)   =  18.10 - 67242/T
C   -log10K(SiO2)  =  15.04 - 66906/T
	AKMG2 = 10**(- 0.94 + 7434/T)

C  MgO(liq) + SiO2(liq) = MgSiO3(liq)
C  log10K(MgSiO3) = - 23.67 + 102856/T
C  -log10K(MgO) = 9.05 - 33621/T
C  -log10K(SiO2) = 15.04 - 66906/T
	AKMG1 = 10**( 0.42 + 2329/T)

C  MgO(liq) + Al2O3(liq) = MgAl2O4(liq)
C  log10K(MgAl2O4) = - 31.55 + 142219/T
C  -log10K(MgO) = 9.05 - 33621/T
C  -log10K(Al2O3) = 23.68 - 108134/T
	AKMG3 = 10**(1.18 + 464/T)

C  MgO(liq) + TiO2(liq) = MgTiO3(liq)
C  log10K(MgTiO3) = - 22.54 + 103180/T
C  -log10K(MgO) = 9.05 - 33621/T
C  -log10K(TiO2) = 13.36 - 66313/T
	AKMG4 = 10**(- 0.13 + 3246/T)

C  MgO(liq) + 2TiO2(liq) = MgTi2O5(liq)
C  log10K(MgTi2O5) = - 35.26 + 169092/T
C  -log10K(MgO) = 9.05 - 33621/T
C  -2log10(TiO2) = 26.72 - 132626/T
	AKMG5 = 10**(0.51 + 2845/T)

C  2MgO(liq) + TiO2(liq) = Mg2TiO4(liq)
C  log10K(Mg2TiO4) = - 30.79 + 137367/T
C  -2log10K(MgO) = 18.10 - 67242/T
C  -log10K(TiO2) = 13.36 - 66313/T
	AKMG6 = 10**(0.67 + 3812/T)

C  3Al2O3(liq) + 2SiO2(liq) = Al6Si2O13(liq)
C  log10K(Al6Si2O13) = - 104.06 + 467589/T
C  -3log10K(Al2O3) = 71.04 - 324402/T
C  -2log10K(SiO2) = 30.08 - 133812/T
	AKAL1 = 10**(- 2.94 + 9375/T)

C  CaO(liq) + Al2O3(liq) = CaAl2O4(liq)
C  log10K(CaAl2O4) = - 33.93 + 154384/T
C  -log10K(CaO) = 8.36 - 36190/T
C  -log10K(Al2O3) = 23.68 - 108134/T
	AKCA1 = 10**(- 1.89 + 10060/T)

C  CaO(liq) + 2Al2O3(liq) = CaAl4O7(liq)
C  log10K(CaAl4O7) = - 56.31 + 262171/T
C  -log10K(CaO) = 8.36 - 36190/T
C  -2log10K(Al2O3) = 47.36 - 216268/T
	AKCA2 = 10**(- 0.59 + 9713/T)

C  12CaO(liq) + 7Al2O3(liq) = Ca12Al14O33(liq)
C  log10K(Ca12Al14O33) = - 272.38 + 1263457/T
C  -12log10K(CaO) = 100.32 - 434280/T
C  -7log10K(Al2O3) = 165.76 - 756938/T
	AKCA3 = 10**(-6.30 + 72239/T)

C  CaO(liq) + SiO2(liq) = CaSiO3(liq)

C  log10K(CaSiO3) = - 22.86 + 108664/T
C  -log10K(CaO) = 8.36 - 36190/T
C  -log10K(SiO2) = 15.04 - 66906/T
	AKCA4 = 10**(0.54 + 5568/T)
C  CaO(liq) + SiO2(liq) = CaSiO3(s)
C  log10K(CaSiO3) = - 19.42 + 93041/T
C  -log10K(CaO) = 8.36 - 36190/T
C  -log10K(SiO2) = 15.04 - 66906/T
C	AKCA4 = 10**(3.98 - 10055/T)

C  CaO(liq) + Al2O3(liq) + 2SiO2(liq) = CaAl2Si2O8(liq)
C  log10K(CaAl2Si2O8) = - 59.49 + 283462/T
C  -log10K(CaO) = 8.36 - 36190/T
C  -log10K(Al2O3) = 23.68 - 108134/T
C  -2log10(SiO2) = 30.08 - 133812/T
	AKCA5 = 10**(+ 2.63 + 5326/T)

C  CaO(liq) + MgO(liq) + 2SiO2(liq) = CaMgSi2O6(liq)
C  log10K(CaMgSi2O6) = -46.03 + 212108/T
C  -log10K(CaO) = 8.36 - 36190/T
C  -log10K(MgO) = 9.05 - 33621/T
C  -2log10K(SiO2) = 30.08 - 133812/T
	AKCA6 = 10**(1.46 + 8485/T)

C  2CaO(liq) + MgO(liq) + 2SiO2(liq) = Ca2MgSi2O7(liq)
C  log10K(Ca2MgSi2O7) = - 55.22 + 255140/T
C  -2log10K(CaO) = 16.72 - 72380/T
C  -log10K(MgO) = 9.05 - 33621/T
C  -2log10K(SiO2) = 30.08 - 133812/T
	AKCA7 = 10**(0.63 + 15327/T)

C  2CaO(liq) + Al2O3(liq) + SiO2(liq) = Ca2Al2SiO7(liq)
C  log10K(Ca2Al2SiO7) = - 53.43 + 258130/T
C  -2log10K(CaO) = 16.72 - 72380/T
C  -log10K(Al2O3) = 23.68 - 108134/T
C  -log10K(SiO2) = 15.04 - 66906/T
	AKCA8 = 10**(2.01 + 10710/T)
C  2CaO(liq) + Al2O3(liq) + SiO2(liq) = Ca2Al2SiO7(s)
C  log10K(Ca2Al2SiO7) = - 54.81 + 252602/T
C  -2log10K(CaO) = 16.72 - 72380/T
C  -log10K(Al2O3) = 23.68 - 108134/T
C  -log10K(SiO2) = 15.04 - 66906/T
C	AKCA8 = 10**(0.63 + 5182/T)

C  CaO(liq) + TiO2(liq) = CaTiO3(liq)
C  log10K(CaTiO3) = - 21.80 + 109558/T
C  -log10K(CaO) = 8.36 - 36190/T
C  -log10K(TiO2) = 13.36 - 66313/T
	AKCA9 = 10**(- 0.08 + 7055/T)

C  2CaO(liq) + SiO2(liq) = Ca2SiO4(liq)
C  log10K(Ca2SiO4) = - 31.13 + 147702/T
C  -2log10K(CaO) = 16.72 - 72380/T
C  -log10K(SiO2) = 15.04 - 66906/T
	AKCA10 = 10**(0.63 + 8416/T)
C  2CaO(liq) + SiO2(liq) = Ca2SiO4(s)
C  log10K(Ca2SiO4) = - 28.24 + 134433/T
C  -2log10K(CaO) = 16.72 - 72380/T
C  -log10K(SiO2) = 15.04 - 66906/T
C	AKCA10 = 10**(3.52 - 4853/T)

C  CaO(liq) + TiO2(liq) + SiO2(liq) = CaTiSiO5(liq)
C  log10K(CaTiSiO5) = - 36.94 + 179480/T
C  -log10K(CaO) = 8.36 - 36190/T
C  -log10K(TiO2) = 13.36 - 66313/T
C  -log10K(SiO2) = 15.04 - 66906/T
	AKCA11 = 10**(- 0.18 + 10071/T)

C  FeO(liq) + TiO2(liq) = FeTiO3(liq)
C  log10K(FeTiO3) = - 22.14 + 100392/T
C  -log10K(FeO) = 8.27 - 30510/T
C  -log10K(TiO2) = 13.36 - 66313/T
	AKFE1 = 10**(- 0.51 + 3569/T)

C  2FeO(liq) + SiO2(liq) = Fe2SiO4(liq)
C  log10K(Fe2SiO4) = - 32.21 + 131029/T
C  -2log10K(FeO) = 16.54 - 61020/T
C  -log10K(SiO2) = 15.04 - 66906/T
	AKFE2 = 10**(- 0.63 + 3103/T)

C  FeO(liq) + Al2O3(liq) = FeAl2O4(liq)
C  log10K(FeAl2O4) = - 33.71 + 144336/T
C  -log10K(FeO) = 8.27 - 30510/T
C  -log10K(Al2O3) = 23.68 - 108134/T
	AKFE3 = 10**(- 1.76 + 5692/T)

C  FeO (liq) + Fe2O3 (liq) = Fe3O4 (liq)
C  Fe3O4 data from Barin 1995
	AKFE4 = 10**(-4.385894544D-1 + 4.3038155175436D03 / T -
     * 3.1050205223386055D6 / T**2.0D0)

C  CaO(liq) + 6Al2O3(liq) = CaAl12O19(liq)
C  log10K(CaAl12O19) = -154.23 + 707606/T
C  -log10K(CaO) = 8.36 - 36190/T
C  -6log10K(Al2O3) = 142.08 - 648804/T
	AKCA12 = 10**(- 3.79 + 22612/T)

C  2MgO(liq) + 2Al2O3(liq) + 5SiO2(liq) = Mg2Al4Si5O18(liq)
C  log10K(Mg2Al4Si5O18) = - 132.38 + 618040/T
C  -2log10K(MgO) = 18.10 - 67242/T
C  -2log10K(Al2O3) = 47.36 - 216268/T
C  -5log10K(SiO2) = 75.20 - 334530/T
	AKMG7 = 10**7.48

C  Na2O(liq) + SiO2(liq) = Na2SiO3(liq)
	AKNA1 = 10**(- 1.33 + 13870/T)
	
C  Na2O(liq) + 2SiO2(liq) = Na2Si2O5(liq)
	AKNA2 = 10**(- 1.39 + 15350/T)
	
C  0.5 Na2O(liq) + 0.5 Al2O3(liq) + SiO2(liq) = NaAlSiO4(liq)
	AKNA3 = 10**(0.65 + 6997/T)

C  0.5 Na2O(liq) + 0.5 Al2O3(liq) + 3SiO2(liq) = NaAlSi3O8(liq)
	AKNA4 = 10**(1.29 + 8788/T)

C  0.5 Na2O(liq) + 0.5 Al2O3(liq) = NaAlO2(liq)
	AKNA5 = 10**(0.55 + 3058/T)

C  Na2O(liq) + TiO2(liq) = Na2TiO3(liq)
	AKNA6 = 10**(- 1.38 + 15445/T)

C  0.5 NA2O(liq) + 0.5 Al2O3(liq) + 2SiO2(liq) = NAAlSi2O6(liq)
	AKNA7 = 10**(- 1.02 + 9607/T)

C  K2O(liq) + SiO2(liq) = K2SiO3(liq)
	AKK1 = 10**(0.2692 + 12735/T)

C  K2O(liq) + 2SiO2(liq) = K2Si2O5(liq)
	AKK2 = 10**(0.3462 + 14685/T)

C  0.5 K2O(liq) + 0.5 Al2O3(liq) + SiO2(liq) = KAlSiO4(liq)
	AKK3 = 10**(0.97 + 8675/T)

C  0.5 K2O(liq) + 0.5 Al2O3(liq) + 3SiO2(liq) = KAlSi3O8(liq)
	AKK4 = 10**(1.11 + 11229/T)

C  0.5 K2O(liq) + 0.5 Al2O3(liq) = KAlO2(liq)
	AKK5 = 10**(0.72 + 4679/T)

C  0.5 K2O(liq) + 0.5 Al2O3(liq) + 2SiO2(liq) = KAlSi2O6(liq)
	AKK6 = 10**(1.53 + 10125/T)

C  K2O(liq) + 4SiO2 (liq) = K2Si4O9 (liq)
        AKK7 = 10**(-0.9648 + 17572 / T)

C  0.5K2O(liq) + CaO(liq) + 0.5Al2O3(liq) + 2SiO2(liq)=KCaAlSi2O7(liq)
	AKK8 = 10**(4.2983 + 17037/T)


C   COMPUTE ACTIVITIES IN THE MELT

1503	COMP = COMP + 1

1500	CONTINUE
	
C   ACTIVITIES OF OXIDES IN THE MELT
	ACSIO2 = FSI * GAMSI
	ACMGO = FMG * GAMMG
	ACFEO = FFE * GAMFE
	
C  The activity of Fe2O3 is estimated using gas chemistry.
C  Then all activities are recomputed.
	IF (COMP .EQ. 1.0D0) THEN
	ACFE2O3 = 0.0D0
	ELSE
	ACFE2O3 = PFE2O3L * GAMFE3
	ENDIF
	
	ACCAO = FCA * GAMCA
	ACAL2O3 = FAL * GAMAL
	ACTIO2 = FTI * GAMTI
	ACNA2O = FNA * GAMNA
	ACK2O = FK * GAMK

C  ACTIVITIES OF COMPLEX SPECIES IN MELT
	ACMG1 = AKMG1 * ACMGO * ACSIO2
	ACMG2 = AKMG2 * ACMGO**2 * ACSIO2
	ACMG3 = AKMG3 * ACMGO * ACAL2O3
	ACMG4 = AKMG4 * ACMGO * ACTIO2
	ACMG5 = AKMG5 * ACMGO * ACTIO2**2
	ACMG6 = AKMG6 * ACMGO**2 * ACTIO2
	ACAL1 = AKAL1 * ACAL2O3**3 * ACSIO2**2
	ACCA1 = AKCA1 * ACCAO * ACAL2O3
	ACCA2 = AKCA2 * ACCAO * ACAL2O3**2
	ACCA3 = AKCA3 * ACCAO**12 * ACAL2O3**7
	ACCA4 = AKCA4 * ACCAO * ACSIO2
	ACCA5 = AKCA5 * ACCAO * ACAL2O3 * ACSIO2**2
	ACCA6 = AKCA6 * ACCAO * ACMGO * ACSIO2**2
	ACCA7 = AKCA7 * ACCAO**2 * ACMGO * ACSIO2**2
	ACCA8 = AKCA8 * ACCAO**2 * ACAL2O3 * ACSIO2
	ACCA9 = AKCA9 * ACCAO * ACTIO2
	ACCA10 = AKCA10 * ACCAO**2 * ACSIO2
	ACCA11 = AKCA11 * ACCAO * ACTIO2 * ACSIO2
	ACFE1 = AKFE1 * ACFEO * ACTIO2
	ACFE2 = AKFE2 * ACFEO**2 * ACSIO2
	ACFE3 = AKFE3 * ACFEO * ACAL2O3
	ACFE4 = AKFE4 * ACFEO * ACFE2O3
	ACCA12 = AKCA12 * ACCAO * ACAL2O3**6
	ACMG7 = AKMG7 * ACMGO**2 * ACAL2O3**2 * ACSIO2**5
	ACNA1 = AKNA1 * ACNA2O * ACSIO2
	ACNA2 = AKNA2 * ACNA2O * ACSIO2**2
	ACNA3 = AKNA3 * DSQRT(ACNA2O) * DSQRT(ACAL2O3) * ACSIO2
	ACNA4 = AKNA4 * DSQRT(ACNA2O) * DSQRT(ACAL2O3) * ACSIO2**3
	ACNA5 = AKNA5 * DSQRT(ACNA2O) * DSQRT(ACAL2O3)
	ACNA6 = AKNA6 * ACNA2O * ACTIO2
	ACNA7 = AKNA7 * DSQRT(ACNA2O) * DSQRT(ACAL2O3) * ACSIO2**2
	ACK1 = AKK1 * ACK2O * ACSIO2
	ACK2 = AKK2 * ACK2O * ACSIO2**2
	ACK3 = AKK3 * DSQRT(ACK2O) * DSQRT(ACAL2O3) * ACSIO2
	ACK4 = AKK4 * DSQRT(ACK2O) * DSQRT(ACAL2O3) * ACSIO2**3
	ACK5 = AKK5 * DSQRT(ACK2O) * DSQRT(ACAL2O3)
	ACK6 = AKK6 * DSQRT(ACK2O) * DSQRT(ACAL2O3) * ACSIO2**2
	ACK7 = AKK7 * ACK2O * (ACSIO2**4)
	ACK8 = AKK8 * DSQRT(ACK2O) * DSQRT(ACAL2O3) * ACCAO * ACSIO2**2



C	Recompute the activity coefficients of the oxides GAM(EL)
C 	from the activities computed above.  
C
C 	GAM(EL) = Activity(pure oxide) / SUM(activities of all
C					  complex melt species
C					  containing the oxide)

	IF (ACSIO2 .NE. 0.0D0) THEN
 	GAMSI = ACSIO2 / (ACMG1 + ACMG2 + ACCA4 + ACCA8 + ACCA10
     * + ACCA11 + ACFE2 + ACSIO2 + ACNA1 + ACNA3 + ACK1 + ACK3
     * + 2.0D0 * (ACAL1 + ACCA5 + ACCA6 + ACCA7 + ACNA2 + ACK2 + ACK6
     * + ACNA7 + ACK8) + 3.0D0 * (ACNA4 + ACK4) + 4.0D0 * ACK7 + 
     *  5.0D0 * ACMG7)
	ELSE
	GAMSI = 0.0D0
	ENDIF
	
	IF (ACMGO .NE. 0.0D0) THEN
	GAMMG = ACMGO / (ACMG1 + ACMG3 + ACMG4 + ACMG5 + ACCA6 + ACCA7
     * + ACMGO + 2.0D0 * (ACMG2 + ACMG6 + ACMG7))
	ELSE
	GAMMG = 0.0D0
	ENDIF
	
	IF (ACFEO .NE. 0.0D0) THEN
	GAMFE = ACFEO / (ACFE1 + 2.0D0 * ACFE2 + ACFE3 + ACFEO + 
     *  2* ACFE2O3 + 3 * ACFE4)
	ELSE
	GAMFE = 0.0D0
	ENDIF
	
C  GAMFE3 is an adjustment factor, not a true activity coefficient because
C  the mole fraction of Fe2O3 in the melt is not known.
	IF (COMP .EQ. 1.0D0) THEN
	GAMFE3 = 1.0D0
	ELSEIF (COMP. NE. 1.0D0 .AND. ACFE2O3 .NE. 0.0D0 .AND. FFE 
     * .NE. 0.0D0) THEN
	GAMFE3 = ACFE2O3 / (ACFE2O3 + ACFE4)
	ELSE
	GAMFE3 = 0.0D0
	ENDIF
	
	IF (ACCAO .NE. 0.0D0) THEN
	GAMCA = ACCAO / (ACCA1 + ACCA2 + ACCA4 + ACCA5 + ACCA6 + ACCA9
     * + ACCA11 + ACCA12 + ACCAO + ACK8 + 2.0D0 * (ACCA7 + ACCA8 + 
     *  ACCA10) + 1.2D1 * ACCA3)
	ELSE
	GAMCA = 0.0D0
	ENDIF
	
	IF (ACAL2O3 .NE. 0.0D0) THEN
	GAMAL = ACAL2O3 / (ACMG3 + ACCA1 + ACCA5 + ACCA8 + ACFE3
     * + ACAL2O3 + 2.0D0 * (ACCA2 + ACMG7) + 3.0D0 * ACAL1
     * + 6.0D0 * ACCA12 + 7.0D0 * ACCA3 + 0.5D0 * (ACNA3 + ACNA4
     * + ACNA5 + ACNA7 + ACK3 + ACK4 + ACK5 + ACK6+ ACK8))
	ELSE
	GAMAL = 0.0D0
	ENDIF
	
	IF (ACTIO2 .NE. 0.0D0) THEN
	GAMTI = ACTIO2 / (ACMG4 + ACMG6 + ACCA9 + ACCA11 + ACFE1
     * + ACTIO2 + ACNA6 + 2.0D0 * ACMG5)
	ELSE
	GAMTI = 0.0D0
	ENDIF
	
	IF (ACNA2O .NE. 0.0D0) THEN
	GAMNA = ACNA2O / (ACNA2O + ACNA1 + ACNA2 + ACNA6
     * + 0.5D0 * (ACNA3 + ACNA4 + ACNA5 + ACNA7))
	ELSE
	GAMNA = 0.0D0
	ENDIF
	
	IF (ACK2O .NE. 0.0D0) THEN
	GAMK = ACK2O / (ACK2O + ACK1 + ACK2 + ACK7 
     * + 0.5D0 * (ACK8 + ACK3 + ACK4 + ACK5 + ACK6))
	ELSE
	GAMK = 0.0D0
	ENDIF



C	RAT(element) is the ratio of the just computed activity to the
C	activity computed in the previous iteration. If RAT(el) = 1,
C 	then the code has converged on a solution for the activity of 
C	that element.

	IF (GAMSI .NE. 0.0D0) THEN
	RATSI = GAMSI/GMSI
	ELSE
	RATSI = 1.0D0
	ENDIF
	
	IF (GAMMG .NE. 0.0D0) THEN
	RATMG = GAMMG/GMMG
	ELSE
	RATMG = 1.0D0
	ENDIF
	
	IF (GAMFE .NE. 0.0D0) THEN
	RATFE = GAMFE/GMFE
	ELSE
	RATFE = 1.0D0
	ENDIF
	
	IF (GAMFE3 .NE. 0.0D0 .AND. FFE .NE. 0.0D0) THEN
	RATFE3 = GAMFE3/GMFE3
	ELSE
	RATFE3 = 1.0D0
	ENDIF
	
	IF (GAMCA .NE. 0.0D0) THEN
	RATCA = GAMCA/GMCA
	ELSE
	RATCA = 1.0D0
	ENDIF
	
	IF (GAMAL .NE. 0.0D0) THEN
	RATAL = GAMAL/GMAL
	ELSE
	RATAL = 1.0D0
	ENDIF
	
	IF (GAMTI .NE. 0.0D0) THEN
	RATTI = GAMTI/GMTI
	ELSE
	RATTI = 1.0D0
	ENDIF
	
	IF (GAMNA .NE. 0.0D0) THEN
	RATNA = GAMNA/GMNA
	ELSE
	RATNA = 1.0D0
	ENDIF
	
	IF (GAMK .NE. 0.0D0) THEN
	RATK = GAMK/GMK
	ELSE
	RATK = 1.0D0
	ENDIF
	
	

C	If RAT(elements) ~ 1, then the code has arrived at a solution 
C	for all of the activities, and moves on to the gas chemistry.
C  	If RAT(elements) does not equal 1, then the activity coefficients
C 	are adjusted and the activities are recomputed until a solution is
C	found.

	IF (DABS(DLOG10(RATSI)) .LT. 1.0D-5 .AND. DABS(DLOG10
     * (RATMG)) .LT. 1.0D-5 .AND. DABS(DLOG10(RATFE)) .LT.
     * 1.0D-5 .AND. DABS(DLOG10(RATCA)) .LT. 1.0D-5 .AND.
     * DABS(DLOG10(RATAL)) .LT. 1.0D-5 .AND. DABS(DLOG10(
     * RATTI)) .LT. 1.0D-5 .AND. DABS(DLOG10(RATNA)) .LT.
     * 1.0D-5 .AND. DABS(DLOG10(RATK)) .LT. 1.0D-5 .AND. 
     * DABS(DLOG10(RATFE3)) .LT. 1.0D-5) GO TO 550

	
C  If the activity calculations have not converged within 30 iterations
C  adjust the activity coefficients using the equations after 1501.
C  Otherwise use the equations directly below (geometric means).
	IF (IIT .GT. 30) GO TO 1501

C  Adjust the activity coefficients by taking the geometric means of the 
C  current values and the values from the previous iteration.
	GAMSI = DSQRT(GAMSI*GMSI)
	GAMMG = DSQRT(GAMMG*GMMG)
	GAMFE = DSQRT(GAMFE*GMFE)
	GAMFE3 = DSQRT(GAMFE3*GMFE3)
	GAMCA = DSQRT(GAMCA*GMCA)
	GAMAL = DSQRT(GAMAL*GMAL)
	GAMTI = DSQRT(GAMTI*GMTI)
	GAMNA = DSQRT(GAMNA*GMNA)
	GAMK = DSQRT(GAMK*GMK)

	GO TO 1502

1501	CONTINUE

	IF (IIT .GT. 5.0D2) GO TO 1505

C  Adjust the activity coefficients 
	GAMSI = (GAMSI * GMSI**2)**(1.0D0/3.0D0)
	GAMMG = (GAMMG * GMMG**2)**(1.0D0/3.0D0)
	GAMFE = (GAMFE * GMFE**2)**(1.0D0/3.0D0)
	GAMFE3 = (GAMFE3 * GMFE3**2)**(1.0D0/3.0D0)
	GAMCA = (GAMCA * GMCA**2)**(1.0D0/3.0D0)
	GAMAL = (GAMAL * GMAL**2)**(1.0D0/3.0D0)
	GAMTI = (GAMTI * GMTI**2)**(1.0D0/3.0D0)
	GAMNA = (GAMNA * GMNA**2)**(1.0D0/3.0D0)
	GAMK = (GAMK * GMK**2)**(1.0D0/3.0D0)
	
	GO TO 1502

1505	GAMSI = (GAMSI * GMSI**4)**(1.0D0/5.0D0)
	GAMMG =(GAMMG * GMMG**4)**(1.0D0/5.0D0)
	GAMFE =(GAMFE * GMFE**4)**(1.0D0/5.0D0)
	GAMFE3 =(GAMFE3 * GMFE3**4)**(1.0D0/5.0D0)
	GAMCA =(GAMCA * GMCA**4)**(1.0D0/5.0D0)
	GAMAL =(GAMAL * GMAL**4)**(1.0D0/5.0D0)
	GAMTI =(GAMTI * GMTI**4)**(1.0D0/5.0D0)
	GAMNA =(GAMNA * GMNA**4)**(1.0D0/5.0D0)
	GAMK =(GAMK * GMK**4)**(1.0D0/5.0D0)
	
	GO TO 1502

1502	CONTINUE

C	save the new values of the activity coefficients
	GMSI = GAMSI
	GMMG = GAMMG
	GMFE = GAMFE
	GMFE3 = GAMFE3
	GMCA = GAMCA
	GMAL = GAMAL
	GMTI = GAMTI
	GMNA = GAMNA
	GMK = GAMK
	

C	WRITE(*,549)IIT
C549	FORMAT(1X,'IIT = ',I5)


C  Count the number of iterations through the activity calculations.
	IIT = IIT + 1
	
C  Return to beginning of activity computations using revised activity 
C  coefficients until convergence upon a solution
 	GO TO 1500


C  Activity calculations are finished. Reset number of iterations for 
C  activity calculations and continue to gas chemistry computations.
550	CONTINUE
	IIT = 0
	
	
C  FORMAT STATEMENTS FOR PRINTING THE ACTIVITY DATA	
548	FORMAT('ACTIVITY COEFFICIENTS (G) OF OXIDES IN THE MELT',/)

551	FORMAT('GSiO2 = ',1PE12.5,/,'GMgO =  ',1PE12.5,/,'GFeO =  ',
     * 1PE12.5,/,'GCaO =  ',1PE12.5,/,'GAl2O3 =',1PE12.5,/,
     * 'GTiO2 = ',1PE12.5,/,'GNa2O = ',1PE12.5,/,'GK2O  = ',1PE12.5,/)
     
547	FORMAT(/,'ACTIVITIES (A) OF SPECIES IN MELT',/)     
     
552    FORMAT('ASiO2',1PE21.6,/,'AMgO',1PE22.6,/,'AFeO',1PE22.6,
     * /,'AFe2O3',1PE20.6,/,'AFe3O4',1PE20.6,/,'ACaO',1PE22.6,/,
     * 'AAl2O3',1PE20.6,/,'ATiO2',1PE21.6,/,'ANa2O',1PE21.6,/,
     * 'AK2O',1PE22.6,/,'AMgSiO3',1PE19.6,/,'AMg2SiO4',1PE18.6,/,
     * 'AMgAl2O4',1PE18.6,/,'AMgTiO3',1PE19.6,/,'AMgTi2O5',1PE18.6,
     * /,'AMg2TiO4',1PE18.6,/,'AAl6Si2O13',1PE16.6,/,'ACaAl2O4',
     * 1PE18.6,/,'ACaAl4O7',1PE18.6,/,'ACa12Al14O33',1PE14.6,/,
     * 'ACaSiO3',1PE19.6,/,'ACaAl2Si2O8',1PE15.6,/,'ACaMgSi2O6',
     * 1PE16.6,/,'ACa2MgSi2O7',1PE15.6,/,'ACa2Al2SiO7',1PE15.6,/,
     * 'ACaTiO3',1PE19.6,/,'ACa2SiO4',1PE18.6,/,'ACaTiSiO5',
     * 1PE17.6,/,'AFeTiO3',1PE19.6,/,'AFe2SiO4',1PE18.6,/,
     * 'AFeAl2O4',1PE18.6,/,'ACaAl12O19',1PE16.6,/,
     * 'AMg2Al4Si5O18',1PE13.6,/,'ANa2SiO3',1PE18.6,/,
     * 'ANa2Si2O5',1PE17.6,/,'ANaAlSiO4',1PE17.6,/,'ANaAlSi3O8'
     * ,1PE16.6,/,'ANaAlO2',1PE19.6,/,'ANa2TiO3',1PE18.6,/,
     * 'ANaAlSi2O6',1PE16.6,/,'AK2SiO3',1PE19.6,/,'AK2Si2O5'
     * ,1PE18.6,/,'AKAlSiO4',1PE18.6,/,'AKAlSi3O8',1PE17.6,/,
     * 'AKAlO2',1PE20.6,/,'AKAlSi2O6',1PE17.6,/,'AK2Si4O9',1PE18.6,
     *  /,'AKCaAlSi2O7',1PE15.6,/)
     
     
     
	AMELTD(1) = ACSIO2
	AMELTD(2) = ACMGO
	AMELTD(3) = ACFEO
	AMELTD(4) = ACCAO
	AMELTD(5) = ACAL2O3
	AMELTD(6) = ACTIO2
	AMELTD(7) = ACNA2O
	AMELTD(8) = ACK2O
	AMELTD(9) = ACMG1
	AMELTD(10) = ACMG2
	AMELTD(11) = ACMG3
	AMELTD(12) = ACMG4
	AMELTD(13) = ACMG5
	AMELTD(14) = ACMG6
	AMELTD(15) = ACAL1
	AMELTD(16) = ACCA1
	AMELTD(17) = ACCA2
	AMELTD(18) = ACCA3
	AMELTD(19) = ACCA4
	AMELTD(20) = ACCA5
	AMELTD(21) = ACCA6
	AMELTD(22) = ACCA7
	AMELTD(23) = ACCA8
	AMELTD(24) = ACCA9
	AMELTD(25) = ACCA10
	AMELTD(26) = ACCA11
	AMELTD(27) = ACFE1
	AMELTD(28) = ACFE2
	AMELTD(29) = ACFE3
	AMELTD(30) = ACCA12
	AMELTD(31) = ACMG7
	AMELTD(32) = ACNA1
	AMELTD(33) = ACNA2
	AMELTD(34) = ACNA3
	AMELTD(35) = ACNA4
	AMELTD(36) = ACNA5
	AMELTD(37) = ACNA6
	AMELTD(38) = ACNA7
	AMELTD(39) = ACK1
	AMELTD(40) = ACK2
	AMELTD(41) = ACK3
	AMELTD(42) = ACK4
	AMELTD(43) = ACK5
	AMELTD(44) = ACK6

	WRITE(3)AMELTD


C  GAS CHEMISTRY CALCULATIONS
C  Calculate gas chemistry in equilibrium with the calculated activities
C  from above. 


C  ADJUST THE ABUNDANCES OF THE MAJOR GASES OF EACH ELEMENT
C  these abundances are used to calculate all other gas chemistry
50	PSIOG = PSIOG * ASIOG
	PO2G = PO2G * AO2G
	PMGOG = PMGOG * AMGOG
	PFEG = PFEG * AFEG
	PCAG = PCAG * ACAG
	PALG = PALG * AALG
	PTIG = PTIG * ATIG
	PNAG = PNAG * ANAG
	PKG = PKG * AKG
	
C  Compute the partial pressures of the vapor species	
C  and activities of the oxides.

C  Ion chemistry:
C    PENEG = P(e-,g), PNACAT = P(Na+,g), PKCAT = P(K+,g)
C    PENEG = PNACAT + PKCAT; 
C    HENCE PENEG = DSQRT((PNACAT + PKCAT)*PENEG)

 	PSIO2L = ESIO2L * PSIOG * DSQRT(PO2G)
	PSIL = ESIL * PSIOG * PO2G**(-0.5)
	PSIG = ESIG * PSIL
	POG = EOG * DSQRT(PO2G)
	PSIO2G = ESIO2G * PSIL * PO2G

	PMGG = EMGG * PMGOG * PO2G**(-0.5)
	PMGOL = EMGOL * PMGG * POG

	PFEOL = EFEOL * PFEG * POG
	PFEL = EFEL * PFEG
	PFEOG = EFEOG * PFEG * DSQRT(PO2G)
	PFE2O3L = EFE2O3L * PFEG**2 * PO2G**1.5
	PFE3O4L = EFE3O4L * PFEG**3 * PO2G**2

	PCAOL = ECAOL * PCAG * POG
	PCAOG = ECAOG * PCAG * DSQRT(PO2G)

	PAL2O3L = EAL2O3L * PALG**2 * POG**3
	PALL = EALL * PALG
	PALOG = EALOG * PALG * DSQRT(PO2G)
	PALO2G = EALO2G * PALG * PO2G
	PAL2OG = EAL2OG * PALG**2 * DSQRT(PO2G)
	PAL2O2G = EAL2O2G * PALG**2 * PO2G

	PTIL = ETIL * PTIG
	PTIOG = ETIOG * PTIL * DSQRT(PO2G)
	PTIO2L = ETIO2L * PTIG * POG**2
	PTIO2G = ETIO2G * PTIL * PO2G

	PNA2OL = ENA2OL * PNAG**2 * POG
	PNAOG = ENAOG * PNAG * POG
	PNA2G = ENA2G * PNAG**2
	PNA2OG = ENA2OG * PNAG**2 * POG

	PK2OL = EK2OL * PKG**2 * POG
	PKOG = EKOG * PKG * POG
	PK2G = EK2G * PKG**2
	PK2OG = EK2OG * PKG**2 * POG

	PENEG = DSQRT(ENACAT*PNAG + EKCAT*PKG)

	IF(PNAG .NE. 0.0D0) THEN
	PNACAT = ENACAT * PNAG / PENEG
	ELSE
	PNACAT = 0.0D0
	ENDIF

	IF(PKG .NE. 0.0D0) THEN
	PKCAT = EKCAT * PKG / PENEG
	ELSE
	PKCAT = 0.0D0
	ENDIF
		

C  Calculate the number density of each species
	CSIOG = PCONV * PSIOG / T
	CO2G = PCONV * PO2G / T
	CSIG = PCONV * PSIG / T
	COG = PCONV * POG / T
	CSIO2G = PCONV * PSIO2G / T
	CMGOG = PCONV * PMGOG / T
	CMGG = PCONV * PMGG / T
	CFEOG = PCONV * PFEOG / T
	CFEG = PCONV * PFEG / T
	CCAG = PCONV * PCAG / T
	CCAOG = PCONV * PCAOG / T
	CALG = PCONV * PALG / T
	CALOG = PCONV * PALOG / T
	CALO2G = PCONV * PALO2G / T
	CAL2OG = PCONV * PAL2OG / T
	CAL2O2G = PCONV * PAL2O2G / T
	CTIOG = PCONV * PTIOG / T
	CTIG = PCONV * PTIG / T
	CTIO2G = PCONV * PTIO2G / T
	CNAG = PCONV * PNAG / T
	CNAOG = PCONV * PNAOG / T
	CNA2G = PCONV * PNA2G / T
	CNA2OG = PCONV * PNA2OG / T
	CKG = PCONV * PKG / T
	CKOG = PCONV * PKOG / T
	CK2G = PCONV * PK2G / T
	CK2OG = PCONV * PK2OG / T
	CENEG = PCONV * PENEG /T
	CNACAT = PCONV * PNACAT / T
	CKCAT = PCONV * PKCAT /T	
	

C	Calculate the number density of each element in the vapor	
	TSI = CSIOG + CSIG + CSIO2G
	TO = 2.0D0 * (CO2G + CSIO2G + CALO2G + CAL2O2G + CTIO2G)
     * + CSIOG + COG + CMGOG + CFEOG + CCAOG + CALOG + CAL2OG 
     * + CTIOG + CNAOG + CNA2OG + CKOG + CK2OG
	TMG = CMGOG + CMGG
	TFE = CFEOG + CFEG
	TCA = CCAOG + CCAG
	TAL = CALG + CALOG + CALO2G + 2.0D0 * (CAL2OG + CAL2O2G)
	TTI = CTIOG + CTIG + CTIO2G
	TNA = CNAG + CNAOG + 2.0D0 * (CNA2G + CNA2OG) + CNACAT
	TK = CKG + CKOG + 2.0D0 * (CK2G + CK2OG) + CKCAT
	


C     RAT = ratio of the number density of O in oxide gases to
C     total O number density
	RAT = (2.0D0 * (TSI + TTI) + TMG + TFE + TCA + 1.5D0 * TAL
     * + 0.5D0 * (TNA + TK)) / TO
	
	
	
C  RECOMPUTE ADJUSTMENT FACTORS FOR THE KEY PRESSURES
C  The activity of the oxides calculated during the activity
C  calculations and from the gas chemistry must agree.
C
C  If the oxide mole fraction for the element is zero, then there
C  should not be any of that element in the vapor (A(element) = 0).
C
C  The O2 abundance is governed by the most abundant oxide in the melt, 
C  normally SiO2.  Once SiO2 is completely vaporized, AO2G is computed 
C  from the remaining species in the melt, in order of volatility.

	IF (PSIO2L .NE. 0.0D0 .AND. FSI .NE. 0.0D0 .AND. GAMSI .NE.
     *  0.0D0) THEN
	ASIOG = 1.0D0 / (RAT * DSQRT(PSIO2L/(FSI * GAMSI)))
	ELSE IF (FSI .EQ. 0.0D0) THEN
	ASIOG = 0.0D0 
	ELSE
	ASIOG = 1.0D0
	ENDIF
	
	IF (PMGOL .NE. 0.0D0) THEN
	AMGOG = FMG * GAMMG / PMGOL
	ELSEIF (FMG .EQ. 0.0D0) THEN
	AMGOG = 0.0D0
	ELSE
	AMGOG = 1.0D0
	ENDIF
	
	IF  (PFEOL .NE. 0.0D0 .OR. PFE2O3L .NE. 0.0D0) THEN
	AFEG = (FFE * GAMFE + ACFE2O3) / (PFEOL + PFE2O3L)
	ELSEIF (FFE .EQ. 0.0D0) THEN
	AFEG = 0.0D0
	ELSE
	AFEG = 1.0D0
	ENDIF
	
	IF(PCAOL.NE.0.0D0.AND.GAMCA.NE.0.0D0.AND.FCA.NE.0.0D0) THEN
	ACAG = 1.0D0/DSQRT(PCAOL/(GAMCA*FCA))
	ELSE IF (FCA.EQ.0.0D0.OR.GAMCA.EQ.0.0D0)THEN
	ACAG=0.0D0
	ELSE
	ACAG=1.0D0
	ENDIF
	
	IF (PAL2O3L .NE. 0.0D0 .AND. GAMAL .NE. 0.0D0 .AND. FAL .NE. 
     * 0.0D0) THEN
	AALG = 1.0D0 / DSQRT(PAL2O3L / (GAMAL * FAL))
	ELSE IF (FAL .EQ. 0.0D0 .OR. GAMAL .EQ. 0.0D0) THEN
	AALG = 0.0D0
	ELSE
	AALG = 1.0D0
	ENDIF
	
	IF (PTIO2L .NE. 0.0D0 .AND. FTI .NE. 0.0D0 .AND. GAMTI .NE.
     * 0.0D0) THEN
	ATIG = 1.0D0 / DSQRT(PTIO2L / (FTI * GAMTI))
	ELSE IF (FTI .EQ. 0.0D0 .OR. GAMTI .EQ. 0.0D0) THEN
	ATIG = 0.0D0
	ELSE
	ATIG = 1.0D0
	ENDIF
	
	IF (PNA2OL .NE. 0.0D0 .AND. FNA .NE. 0.0D0 .AND. GAMNA .NE.
     * 0.0D0) THEN
	ANAG = 1.0D0 / DSQRT(PNA2OL/(FNA * GAMNA))
	ELSE IF (FNA .EQ. 0.0D0) THEN
	ANAG = 0.0D0
	ELSE
	ANAG = 1.0D0
	ENDIF
	
	IF (PK2OL .NE. 0.0D0 .AND. FK .NE. 0.0D0 .AND. GAMK .NE.
     * 0.0D0) THEN
	AKG = 1.0D0 / DSQRT(PK2OL/(FK * GAMK))
	ELSE IF (FK .EQ. 0.0D0) THEN
	AKG = 0.0D0
	ELSE
	AKG = 1.0D0
	ENDIF
	
C  Adjustment factor for oxygen is governed by the most abundant volatile
C  metal oxide present in the melt.
	IF (PSIO2L .NE. 0.0D0 .AND. FSI .NE. 0.0D0)THEN
	AO2G = RAT * FSI * GAMSI / PSIO2L
	ELSE IF (FSI .EQ. 0.0D0 .AND. FMG .NE. 0.0D0)THEN
	AO2G = RAT * FMG*GAMMG / PMGOL
	ELSE IF (FSI .EQ. 0.0D0 .AND. FMG .EQ. 0.0D0 .AND. FFE .NE. 
     * 0.0D0) THEN
        AO2G = RAT * (FFE * GAMFE + ACFE2O3) / (PFEOL + PFE2O3L)
        ELSE IF (FSI .EQ. 0.0D0 .AND. FMG .EQ. 0.0D0 .AND. FFE .EQ.
     * 0.0D0 .AND. FCA .NE. 0.0D0) THEN
        AO2G = RAT * FCA * GAMCA / PCAOL
        ELSE IF (FSI .EQ. 0.0D0 .AND. FMG .EQ. 0.0D0 .AND. FFE .EQ. 
     * 0.0D0 .AND. FCA .EQ. 0.0D0 .AND. FAL .NE. 0.0D0) THEN
        AO2G = RAT * FAL * GAMAL / PAL2O3L
        ELSE IF (FSI .EQ. 0.0D0 .AND. FMG .EQ. 0.0D0 .AND. FFE .EQ.
     * 0.0D0 .AND. FCA .EQ. 0.0D0 .AND. FAL .EQ. 0.0D0 .AND. FTI .NE. 
     * 0.0D0) THEN
        AO2G = RAT * FTI * GAMTI / PTIO2L
        ELSE IF (FSI .EQ. 0.0D0 .AND. FMG .EQ. 0.0D0 .AND. FFE .EQ.
     * 0.0D0 .AND. FCA .EQ. 0.0D0 .AND. FAL .EQ. 0.0D0 .AND. FTI .EQ.
     * 0.0D0 .AND. FNA .NE. 0.0D0) THEN
        AO2G = RAT * FNA * GAMNA / PNA2OL
        ELSE IF (FSI .EQ. 0.0D0 .AND. FMG .EQ. 0.0D0 .AND. FFE .EQ.
     * 0.0D0 .AND. FCA .EQ. 0.0D0 .AND. FAL .EQ. 0.0D0 .AND. FTI .EQ.
     * 0.0D0 .AND. FNA .EQ. 0.0D0 .AND. FK .NE. 0.0D0) THEN
     	AO2G = RAT * FK * GAMK / PK2OL
        ELSE
        AO2G = 1.0D0
        ENDIF
	
	
	
C	WRITE(1,4)ASIOG,AO2G,AMGOG,AFEG,ACAG,AALG,ATIG,ANAG,AKG
C4	FORMAT(1X,'ASiOG,AO2G,AMgOG,AFeG,ACaG,AAlG,ATiG,ANaG,AKG = '
C     *  ,/,1PE13.5,1PE13.5,1PE13.5,1PE13.5,/,1PE13.5,1PE13.5,1PE13.5,
C     *  1PE13.5,/,1PE13.5,/)
C	PAUSE

C	PROD = ASIOG*AO2G*AMGOG*AFEG*ACAG*AALG*ATIG*ANAG*AKG
C	WRITE(*,6)PROD
C6	FORMAT(1X,1PD15.7)




C	If the adjutment factors for the pressures are NOT ~1, or 0, then 
C	the factors are recomputed and gas chemistry is repeated (50) 
C  	until a solution is converged upon. Otherwise the code continues
C 	to the next computations (80).

	IF (((ASIOG .LT. 1.00000230259D0 .AND. ASIOG .GT. 
     *  9.99997697418D-1) .OR. ASIOG .EQ. 0.0D0).AND. ((AO2G .LT. 
     *  1.00000230259D0 .AND. AO2G .GT. 9.99997697418D-1) .OR. AO2G 
     *  .EQ. 0.0D0) .AND. ((AMGOG .LT. 1.00000230259D0 .AND. AMGOG 
     *  .GT. 9.99997697418D-1) .OR. AMGOG .EQ. 0.0D0) .AND. ((AFEG 
     *  .LT. 1.00000230259D0 .AND. AFEG .GT. 9.99997697418D-1) .OR. 
     *  AFEG .EQ. 0.0D0) .AND. ((ACAG .LT. 1.00000230259D0 .AND. ACAG 
     *  .GT. 9.99997697418D-1) .OR. ACAG .EQ. 0.0D0) .AND. ((AALG .LT.
     *  1.00000230259D0 .AND. AALG .GT. 9.99997697418D-1) .OR. AALG 
     *  .EQ. 0.0D0) .AND. ((ATIG .LT. 1.00000230259D0 .AND. ATIG .GT. 
     *  9.99997697418D-1) .OR. ATIG .EQ. 0.0D0) .AND. ((ANAG .LT. 
     *  1.00000230259D0 .AND. ANAG .GT. 9.99997697418D-1) .OR. ANAG 
     *  .EQ. 0.0D0) .AND. ((AKG .LT. 1.00000230259D0 .AND. AKG 
     *  .GT. 9.99997697418D-1) .OR. AKG .EQ. 0.0D0)) THEN 
        GOTO 80
	ELSE
	GOTO 50
	ENDIF


80	CONTINUE


C  If this is the first run through activity/gas calculations for this step
C  then go back to activity calculations and add in the iron oxides
C  Fe2O3 and Fe3O4. Repeat activity and gas calculations until answers converge.
	IF (COMP .EQ. 1.0D0) GOTO 1503


C  Reset the counting factor COMP for the next vaporization step	
	COMP = 0.0D0
	
	
	
199	FORMAT(/,'GAS PARTIAL PRESSURES (P) IN VAPOR',/)

200	FORMAT(/,'PO = ',1PE17.6,/,'PO2 = ',1PE16.6,/,'PMg = ',1PE16.6,
     * /,'PMgO = ',1PE15.6,/,'PSi = ',1PE16.6,/,'PSiO = ',1PE15.6,/,
     * 'PSiO2 = ',1PE14.6,/,'PFe = ',1PE16.6,/,'PFeO = ',1PE15.6,/,
     * 'PAl = ',1PE16.6,/,'PAlO = ',1PE15.6,/,'PAlO2 = ',1PE14.6,/,
     * 'PAl2O = ',1PE14.6,/,'PAl2O2 = ',1PE13.6,/,'PCa = ',1PE16.6,/,
     * 'PCaO = ',1PE15.6,/,'PNa = ',1PE16.6,/,'PNa2 = ',1PE15.6,/,
     * 'PNaO = ',1PE15.6,/,'PNa2O = ',1PE14.6,/,'PNa+ = ',1PE15.6,/,
     * 'PK = ',1PE17.6,/,'PK2 = ',1PE16.6,/,'PKO = ',1PE16.6,/,
     * 'PK2O = ',1PE15.6,/,'PK+ = ',1PE16.6,/,'PTi = ',1PE16.6,/,
     * 'PTiO = ',1PE15.6,/,'PTiO2 = ',1PE14.6,/,'Pe- = ',1PE16.6,/)
     
     
     
     
C       CALCULATION OF TOTAL O GAS PRESSURE (PTO)

	PTO = POG + PO2G

C	CALCULATION OF TOTAL SI GAS PRESSURE (PTSI)

	PTSI = PSIG + PSIOG + PSIO2G
	
C	CALCULATION OF TOTAL MG GAS PRESSURE (PTMG)

	PTMG = PMGG + PMGOG
	
C	CALCULATION OF TOTAL FE GAS PRESSURE (PTFE)

	PTFE = PFEG + PFEOG
	
C	CALCULATION OF TOTAL CA GAS PRESSURE (PTCA)

	PTCA = PCAG + PCAOG
	
C	CALCULATION OF TOTAL AL GAS PRESSURE (PTAL)

	PTAL = PALG + PALOG + PALO2G + PAL2OG + PAL2O2G
	
C	CALCULATION OF TOTAL NA GAS PRESSURE (PTNA)

	PTNA = PNAG + PNA2G + PNAOG + PNA2OG + PNACAT
	
C	CALCULATION OF TOTAL K GAS PRESSURE (PTK)

	PTK = PKG + PK2G + PKOG + PK2OG + PKCAT
	
C	CALCULATION OF TOTAL TI GAS PRESSURE (PTTI)

	PTTI = PTIG + PTIOG + PTIO2G
	
C	CALCULATION OF TOTAL GAS PRESSURE (PTOT)

	PTOT=PTO + PTSI + PTMG + PTFE + PTCA + PTAL + PTNA + PTK + PTTI
     * + PENEG
     
     
	
C	FORMAT STATEMENT TO PRINT OUT THE TOTAL PRESSURE	
	
610	FORMAT('PTOT = ',1PE16.7,/)



C	CALCULATE GAS MOLE FRACTIONS (P/PTOT)

	XO = POG/PTOT
	XO2 = PO2G/PTOT
	
	XMG = PMGG/PTOT
	XMGO = PMGOG/PTOT
	
	XSI = PSIG/PTOT
	XSIO = PSIOG/PTOT
	XSIO2 = PSIO2G/PTOT
		
	XFE = PFEG/PTOT
	XFEO = PFEOG/PTOT
	
	XAL = PALG/PTOT
	XALO = PALOG/PTOT
	XALO2 = PALO2G/PTOT
	XAL2O = PAL2OG/PTOT
	XAL2O2 = PAL2O2G/PTOT
	
	XCA = PCAG/PTOT
	XCAO = PCAOG/PTOT
	
	XNA = PNAG/PTOT
	XNA2 = PNA2G/PTOT
	XNAO = PNAOG/PTOT
	XNA2O = PNA2OG/PTOT
	XNACAT = PNACAT/PTOT
	
	XK = PKG/PTOT
	XK2 = PK2G/PTOT
	XKO = PKOG/PTOT
	XK2O = PK2OG/PTOT
	XKCAT = PKCAT/PTOT
	
	XTI = PTIG/PTOT
	XTIO = PTIOG/PTOT
	XTIO2 = PTIO2G/PTOT
	
	XENEG = PENEG/PTOT
	
	
	
C	FORMAT STATEMENTS TO PRINT OUT GAS MOLE FRACTIONS

620	FORMAT(/,'GAS MOLE FRACTIONS (X) IN VAPOR',/)
	
201	FORMAT(/,'XO = ',1PE17.6,/,'XO2 = ',1PE16.6,/,'XMg = ',1PE16.6,
     * /,'XMgO = ',1PE15.6,/,'XSi = ',1PE16.6,/,'XSiO = ',1PE15.6,/,
     * 'XSiO2 = ',1PE14.6,/,'XFe = ',1PE16.6,/,'XFeO = ',1PE15.6,/,
     * 'XAl = ',1PE16.6,/,'XAlO = ',1PE15.6,/,'XAlO2 = ',1PE14.6,/,
     * 'XAl2O = ',1PE14.6,/,'XAl2O2 = ',1PE13.6,/,'XCa = ',1PE16.6,/,
     * 'XCaO = ',1PE15.6,/,'XNa = ',1PE16.6,/,'XNa2 = ',1PE15.6,/,
     * 'XNaO = ',1PE15.6,/,'XNa2O = ',1PE14.6,/,'XNa+ = ',1PE15.6,/,
     * 'XK = ',1PE17.6,/,'XK2 = ',1PE16.6,/,'XKO = ',1PE16.6,/,
     * 'XK2O = ',1PE15.6,/,'XK+ = ',1PE16.6,/,'XTi = ',1PE16.6,/,
     * 'XTiO = ',1PE15.6,/,'XTiO2 = ',1PE14.6,/,'Xe- = ',1PE16.6,/)
     
     
C  Calculate the total mole fraction of an element in the gas
	TTOT = TSI + TMG + TFE + TCA + TAL + TTI + TNA + TK
	ATMSI = TSI / TTOT
	ATMMG = TMG / TTOT
	ATMFE = TFE / TTOT
	ATMCA = TCA / TTOT
	ATMAL = TAL / TTOT
	ATMTI = TTI / TTOT
	ATMNA = TNA / TTOT
	ATMK = TK / TTOT
	
	
	AMDATA(8) = ATMSI
	AMDATA(9) = ATMMG
	AMDATA(10) = ATMFE
	AMDATA(11) = ATMCA
	AMDATA(12) = ATMAL
	AMDATA(13) = ATMTI
	AMDATA(18) = ATMNA
	AMDATA(19) = ATMK
	
	WRITE(2)AMDATA
	
	
	
C  If this is the very first computation (step 0), print the  
C  calculated equilibrium abundances before removing mass for the 
C  first vaporization step.
	IF (IFIRST .EQ. 0) GO TO 1000
C  Now remove mass from vaporization.
1001	IFIRST = 1



600	FORMAT(1X,' TSI = ',1PE13.5,'  TMG = ',1PE13.5,'  TFE = ',
     * 1PE13.5,/,'  TCA = ',1PE13.5,'  TAL = ',1PE13.5,'  TTI = ',
     * 1PE13.5,/,'  TNA = ',1PE13.5,'  TK  = ',1PE13.5,/)

500	FORMAT(1X,' ATMSI = ',1PE13.5,'  ATMMG = ',1PE13.5,
     * '  ATMFE = ',1PE13.5,/,'  ATMCA = ',1PE13.5,'  ATMAL = ',
     * 1PE13.5,'  ATMTI = ',1PE13.5,/,'  ATMNA = ',1PE13.5,
     * '  ATMK  = ',1PE13.5,/)



C	Compute the step size.  The most volatile element in the melt will be 
C	reduced by 5%.  The computation is done for both the mole fractions of
C	the elements and the atomic abundances because once PLAN(element) becomes
C	smaller than ~1D-35, it gets put to zero due to the limitations of FORTRAN.
C	The mole fractions are renormalized below and are used to compute the 
C	oxide mole fractions.  The PLAN(element) values are 
C	used to compute the fraction vaporized.

	ATMAX = 0.0D0
	IF (CONSI .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMSI/CONSI)
	IF (CONMG .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMMG/CONMG)
	IF (CONFE .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMFE/CONFE)
	IF (CONCA .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMCA/CONCA)
	IF (CONAL .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMAL/CONAL)
	IF (CONTI .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMTI/CONTI)
	IF (CONNA .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMNA/CONNA)
	IF (CONK .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMK/CONK)
	FACT = 0.05D0/ATMAX

	ATMAX = 0.0D0
	IF (CONSI .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMSI/PLANSI)
  	IF (CONMG .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMMG/PLANMG)
  	IF (CONFE .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMFE/PLANFE)
  	IF (CONCA .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMCA/PLANCA)
  	IF (CONAL .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMAL/PLANAL)
  	IF (CONTI .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMTI/PLANTI)
  	IF (CONNA .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMNA/PLANNA)
  	IF (CONK .GT. 1.0D-20) ATMAX = DMAX1(ATMAX,ATMK/PLANK)
	FACT1 = 0.05D0/ATMAX

C  Compute the new abundance of each element to be used in the next
C  vaporization step. CON(El) = mole fraction in melt
C  Plan(El) = total atomic abundance
C  if plan(el) = 0, then set con(el) = 0 also.

	CONSI = CONSI - FACT * ATMSI
	CONMG = CONMG - FACT * ATMMG
	CONFE = CONFE - FACT * ATMFE
	CONCA = CONCA - FACT * ATMCA
	CONAL = CONAL - FACT * ATMAL
	CONTI = CONTI - FACT * ATMTI
	CONNA = CONNA - FACT * ATMNA
	CONK = CONK - FACT * ATMK
						
	PLANSI = PLANSI - FACT1 * ATMSI
	IF (PLANSI .LT. 0.0D0) PLANSI = 0.0D0
	IF (PLANSI .EQ. 0.0D0) CONSI = 0.0D0
	PLANMG = PLANMG - FACT1 * ATMMG
	IF (PLANMG .LT. 0.0D0) PLANMG = 0.0D0
	IF (PLANMG .EQ. 0.0D0) CONMG = 0.0D0
	PLANFE = PLANFE - FACT1 * ATMFE
	IF (PLANFE .LT. 2.0D-20) PLANFE = 0.0D0
	IF (PLANFE .EQ. 0.0D0) CONFE = 0.0D0
	PLANCA = PLANCA - FACT1 * ATMCA
	IF (PLANCA .LT. 0.0D0) PLANCA = 0.0D0
	IF (PLANCA .EQ. 0.0D0) CONCA = 0.0D0
	PLANAL = PLANAL - FACT1 * ATMAL
	IF (PLANAL .LT. 0.0D0) PLANAL = 0.0D0
	IF (PLANAL .EQ. 0.0D0) CONAL = 0.0D0
	PLANTI = PLANTI - FACT1 * ATMTI
	IF (PLANTI .LT. 0.0D0) PLANTI = 0.0D0
	IF (PLANTI .EQ. 0.0D0) CONTI = 0.0D0
	PLANNA = PLANNA - FACT1 * ATMNA
	IF (PLANNA .LT. 0.0D0) PLANNA = 0.0D0
	IF (PLANNA .EQ. 0.0D0) CONNA = 0.0D0
	PLANK = PLANK - FACT1 * ATMK
	IF (PLANK .LT. 0.0D0) PLANK = 0.0D0
	IF (PLANK .EQ. 0.0D0) CONK = 0.0D0
	PLMANT = PLANSI + PLANMG + PLANFE + PLANCA + 
     * PLANAL + PLANTI + PLANNA + PLANK
	PLANRAT = PLMANT / PLMAN0
	
	
	
C	CALCULATE FRACTION OF MAGMA THAT IS VAPORIZED

	VAP = 1 - PLANRAT
	
C	CALCULATE WEIGHT PERCENT VAPORIZED

	MASS1 = PLANSI * MWSiO2 + PLANMG * MWMgO + PLANFE * MWFeO + 
     * PLANCA * MWCaO +  
     * PLANAL * 0.5 * MWAl2O3 + PLANTI * MWTiO2 + 
     * 0.5 * PLANNA * MWNa2O + 0.5 * PLANK * MWK2O
	
	WEIGHT = (MASS0-MASS1)/MASS0




202	FORMAT(/,'FRACTION OF MAGMA THAT IS VAPORIZED',/)

203	FORMAT('VAP = ',1PE16.8,/,'WTVAP = ',1PE16.8,/)

320	FORMAT(/, 1X,'ATOMIC ABUNDANCES ON COSMOCHEMICAL SCALE',/)

330	FORMAT(1X,' ASi   = ',1PE16.9,/,'  AMg   = ',1PE16.9,/,
     * '  AFe   = ',1PE16.9,/,'  ACa   = ',1PE16.9,/,'  AAl   = ',
     * 1PE16.9,/,'  ATi   = ',1PE16.9,/,'  ANa   = ',1PE16.9,/,
     * '  AK    = ',1PE16.9,/)



C	renormalize the abundances
C	CONTOT used to compute the mole fractions of the elements
C	CONTOT1 used to compute the oxide mole fractions

	CONTOT = CONSI + CONMG + CONFE + CONCA + CONAL + CONTI 
     * + CONNA + CONK
	CONTOT1 = CONSI + CONMG + CONFE + CONCA + 
     * 0.5D0*CONAL + CONTI + 0.5D0 * (CONNA + CONK)
     
     

C	relative abundances of the metals (by molecule)
	FSI = CONSI / CONTOT1
	FMG = CONMG / CONTOT1
	FFE = CONFE / CONTOT1
	FCA = CONCA / CONTOT1
	FAL = 0.5D0 * CONAL / CONTOT1
	FTI = CONTI / CONTOT1
	FNA = 0.5D0 * CONNA /CONTOT1
	FK = 0.5D0 * CONK / CONTOT1

C	relative abundances of the metals (by atom)
	CONSI = CONSI / CONTOT
	CONMG = CONMG / CONTOT
	CONFE = CONFE / CONTOT
	CONCA = CONCA / CONTOT
	CONAL = CONAL / CONTOT
	CONTI = CONTI / CONTOT
	CONNA = CONNA / CONTOT
	CONK = CONK / CONTOT
	
	

C  The counting factor IPRN decides how many vaporization steps
C  are printed in the output file, e.g., every step, every 5 steps, 
C  every 20 steps, etc.  The factor ISTEP counts the number of times
C  mass has been removed from the system (=vaporization step). 
	IPRN = IPRN + 1
	ISTEP = ISTEP + 1
	IF (IPRN .NE. 5) GOTO 2000

C  Reset IPRN and print equilibrium results to output file. 
	IPRN = 0
	
	
1000	WRITE(*,270)ISTEP
	WRITE(1,270)ISTEP
270	FORMAT('STEP NUMBER ',I4,/)



C	NOW WRITE THE OXIDE MOLE FRACTIONS IN THE MELT

	WRITE(1,360)
	WRITE(1,400)FSI,FMG,FFE,FCA,FAL,FTI,FNA,FK
	
C	NOW WRITE THE ACTIVITY COEFFICIENTS FOR EACH METAL

	WRITE(1,548)
	WRITE(*,551)GAMSI,GAMMG,GAMFE,GAMCA,GAMAL,GAMTI,
     * GAMNA,GAMK
	WRITE(1,551)GAMSI,GAMMG,GAMFE,GAMCA,GAMAL,GAMTI,
     * GAMNA,GAMK

	WRITE(*,552)ACSIO2,ACMGO,ACFEO,ACFE2O3,ACFE4,ACCAO,ACAL2O3,
     * ACTIO2,ACNA2O,ACK2O,ACMG1,ACMG2,ACMG3,ACMG4,ACMG5,ACMG6,ACAL1,
     * ACCA1,ACCA2,ACCA3,ACCA4,ACCA5,ACCA6,ACCA7,ACCA8,ACCA9,ACCA10,
     * ACCA11,ACFE1,ACFE2,ACFE3,ACCA12,ACMG7,ACNA1,ACNA2,ACNA3,ACNA4,
     * ACNA5,ACNA6,ACNA7,ACK1,ACK2,ACK3,ACK4,ACK5,ACK6,ACK7,ACK8
     
     	WRITE(1,547)
     	
	WRITE(1,552)ACSIO2,ACMGO,ACFEO,ACFE2O3,ACFE4,ACCAO,ACAL2O3,
     * ACTIO2,ACNA2O,ACK2O,ACMG1,ACMG2,ACMG3,ACMG4,ACMG5,ACMG6,ACAL1,
     * ACCA1,ACCA2,ACCA3,ACCA4,ACCA5,ACCA6,ACCA7,ACCA8,ACCA9,ACCA10,
     * ACCA11,ACFE1,ACFE2,ACFE3,ACCA12,ACMG7,ACNA1,ACNA2,ACNA3,ACNA4,
     * ACNA5,ACNA6,ACNA7,ACK1,ACK2,ACK3,ACK4,ACK5,ACK6,ACK7,ACK8

	WRITE(*,200)POG,PO2G,PMGG,PMGOG,PSIG,PSIOG,PSIO2G,PFEG,PFEOG,
     * PALG,PALOG,PALO2G,PAL2OG,PAL2O2G,PCAG,PCAOG,PNAG,PNA2G,PNAOG,
     * PNA2OG,PNACAT,PKG,PK2G,PKOG,PK2OG,PKCAT,PTIG,PTIOG,PTIO2G,PENEG
     
	WRITE(1,199)
	
C	NOW WRITE THE GAS PARTIAL PRESSURES IN THE VAPOR	
	
	WRITE(1,200)POG,PO2G,PMGG,PMGOG,PSIG,PSIOG,PSIO2G,PFEG,PFEOG,
     * PALG,PALOG,PALO2G,PAL2OG,PAL2O2G,PCAG,PCAOG,PNAG,PNA2G,PNAOG,
     * PNA2OG,PNACAT,PKG,PK2G,PKOG,PK2OG,PKCAT,PTIG,PTIOG,PTIO2G,PENEG
     
C	NOW WRITE THE TOTAL VAPOR PRESSURE

     	WRITE(1,610)PTOT
     	
     	WRITE(1,620)
     	
C	NOW WRITE THE GAS MOLE FRACTIONS IN THE VAPOR     	
     	
     	WRITE(1,201)XO,XO2,XMG,XMGO,XSI,XSIO,XSIO2,XFE,XFEO,XAL,XALO,
     * XALO2,XAL2O,XAL2O2,XCA,XCAO,XNA,XNA2,XNAO,XNA2O,XNACAT,XK,XK2,
     * XKO,XK2O,XKCAT,XTI,XTIO,XTIO2,XENEG
     
C	NOW WRITE THE FRACTION OF MAGMA VAPORIZED

	WRITE(1,202)
	
	WRITE(1,203)VAP,WEIGHT

	WRITE(*,600)TSI,TMG,TFE,TCA,TAL,TTI,TNA,TK
	WRITE(1,600)TSI,TMG,TFE,TCA,TAL,TTI,TNA,TK
	WRITE(*,500)ATMSI,ATMMG,ATMFE,ATMCA,ATMAL,ATMTI,ATMNA,ATMK
	WRITE(1,500)ATMSI,ATMMG,ATMFE,ATMCA,ATMAL,ATMTI,ATMNA,ATMK
     
     	WRITE(1,409)

	WRITE(1,410)CONSI,CONMG,CONFE,CONCA,CONAL,CONTI,
     * CONNA,CONK
	WRITE(*,410)CONSI,CONMG,CONFE,CONCA,CONAL,CONTI,
     * CONNA,CONK

	WRITE(1,320)
	WRITE(1,330)PLANSI,PLANMG,PLANFE,PLANCA,PLANAL,PLANTI,
     * PLANNA,PLANK
	WRITE(*,330)PLANSI,PLANMG,PLANFE,PLANCA,PLANAL,PLANTI,
     * PLANNA,PLANK
	
	
	
	IF (IFIRST .EQ. 0) GO TO 1001
2000	CONTINUE

	
	AMDATA(1) = PLANRAT
	AMDATA(2) = CONSI
	AMDATA(3) = CONMG
	AMDATA(4) = CONFE
	AMDATA(5) = CONCA
	AMDATA(6) = CONAL
	AMDATA(7) = CONTI
	AMDATA(16) = CONNA
	AMDATA(17) = CONK



C  Limit the number of steps to calculate. 
	IF (IREP .EQ. 5001) GO TO 3000
	IREP = IREP + 1

C	1503 goes back to the calculation of the activities
	GOTO 1503
	
	
	
3000	CONTINUE
	AMDATA(8) = ATMSI
	AMDATA(9) = ATMMG
	AMDATA(10) = ATMFE
	AMDATA(11) = ATMCA
	AMDATA(12) = ATMAL
	AMDATA(13) = ATMTI
	AMDATA(18) = ATMNA
	AMDATA(19) = ATMK

	WRITE(2)AMDATA

	CLOSE(1)
	CLOSE(2)
	CLOSE(3)
	STOP
	END
	
	

C Melting points of oxides
C 	Oxide			m.p. (Kelvin)	Note or Reference
C 	Na2O			1405
C 	K2O			1175		Glushko 2nd ed vol. 4
C	Al2O3			2327		corundum
C	SiO2			1996		cristobalite
C	TiO2			2130±20		rutile
C	CaO			3172±20		Gurvich 3rd ed vol. 3
C	MgO			3100		Gurvich 3rd ed vol.3
C	FeO			1650 		JANAF 4th ed
C	Fe2O3			1895		Robie & Hemingway
C	Fe3O4			1870		JANAF 4th ed.

C Thermodynamic data & reactions used to make the coefficients above
C from my 2-2-86 calculations
C log K = A + B/T

C 0.5 Na2O(s) + 0.5 Al2O3(corundum) + SiO2(cristobalite) = NaAlSiO4(nepheline)
C 0.61 + 7357/T		400-1500 K	0.99986
C Robie - nepheline; JANAF - oxides

C 0.5 K2O(s) + 0.5 Al2O3(corundum) + SiO2(cristobalite) = KAlSiO4(s)
C 0.44 + 10,354/T	400-1800 K	0.99996
C Robie - KAlSiO4; JANF - oxides