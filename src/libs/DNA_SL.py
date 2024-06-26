# written in SantaLucia notation: ATTA=AT/TA ,
# 5'-AT-3' / 3'-TA-5'
dH_stack = {
 'AATT': -7.6,
 'ATTA': -7.2,
 'TAAT': -7.2,
 'CAGT': -8.5,
 'GTCA': -8.4,
 'CTGA': -7.8,
 'GACT': -8.2,
 'CGGC': -10.6,
 'GCCG': -9.8,
 'GGCC': -8.0,
 'TTAA': -7.6,
 'TGAC': -8.5,
 'ACTG': -8.4,
 'AGTC': -7.8,
 'TCAG': -8.2,
 'CCGG': -8.0,
 # AC mis
 'AATC': 2.3,
 'ACTA': 5.3,
 'CAGC': 1.9,
 'CCGA': 0.6,
 'GACC': 5.2,
 'GCCA': -0.7,
 'TAAC': 3.4,
 'TCAA': 7.6,
 'CTAA': 2.3,
 'AACT': 7.6,
 'ATCA': 5.3,
 'CAAT': 3.4,
 'CGAC': 1.9,
 'ACCG': -0.7,
 'AGCC': 0.6,
 'CCAG': 5.2,
 # GA mis
 'AATG': -0.6,
 'AGTA': -0.7,
 'CAGG': -0.7,
 'CGGA': -4.0,
 'GACG': -0.6,
 'GGCA': 0.5,
 'TAAG': 0.7,
 'TGAA': 3.0,
 'GTAA': -0.6,
 'AAGT': 3.0,
 'ATGA': -0.7,
 'GAAT': 0.7,
 'GGAC': -0.7,
 'ACGG': 0.5,
 'AGGC': -4.0,
 'GCAG': -0.6,
 # CT mis
 'ACTT': 0.7,
 'ATTC': -1.2,
 'CCGT': -0.8,
 'CTGC': -1.5,
 'GCCT': 2.3,
 'GTCC': 5.2,
 'TCAT': 1.2,
 'TTAC': 1.0,
 'TTCA': 0.7,
 'CATT': 1.0,
 'CTTA': -1.2,
 'TACT': 1.2,
 'TGCC': -0.8,
 'CCTG': 5.2,
 'CGTC': -1.5,
 'TCCG': 2.3,
 # GT mis
 'AGTT': 1.0,
 'ATTG': -2.5,
 'CGGT': -4.1,
 'CTGG': -2.8,
 'GGCT': 3.3,
 'GGTT': 5.8,
 'GTCG': -4.4,
 'GTTG': 4.1,
 'TGAT': -0.1,
 'TGGT': -1.4,
 'TTAG': -1.3,
 'TTGA': 1.0,
 'GATT': -1.3,
 'GTTA': -2.5,
 'TAGT': -0.1,
 'TGGC': -4.1,
 'GCTG': -4.4,
 'GGTC': -2.8,
 'TCGG': 3.3,
 'TTGG': 5.8,
 # AA mis
 'AATA': 1.2,
 'CAGA': -0.9,
 'GACA': -2.9,
 'TAAA': 4.7,
 'ATAA': 1.2,
 'AAAT': 4.7,
 'AGAC': -0.9,
 'ACAG': -2.9,
 # CC mis
 'ACTC': 0.0,
 'CCGC': -1.5,
 'GCCC': 3.6,
 'TCAC': 6.1,
 'CTCA': 0.0,
 'CACT': 6.1,
 'CGCC': -1.5,
 'CCCG': 3.6,
 # GG mis
 'AGTG': -3.1,
 'CGGG': -4.9,
 'GGCG': -6.0,
 'TGAG': 1.6,
 'GTGA': -3.1,
 'GAGT': 1.6,
 'GGGC': -4.9,
 'GCGG': -6.0,
 # TT mis
 'ATTT': -2.7,
 'CTGT': -5.0,
 'GTCT': -2.2,
 'TTAT': 0.2,
 'TTTA': -2.7,
 'TATT': 0.2,
 'TGTC': -5.0,
 'TCTG': -2.2
}
dS_stack = {
 'AATT': -21.3,
 'ATTA': -20.4,
 'TAAT': -20.4,
 'CAGT': -22.7,
 'GTCA': -22.4,
 'CTGA': -21.0,
 'GACT': -22.2,
 'CGGC': -27.2,
 'GCCG': -24.4,
 'GGCC': -19.9,
 'TTAA': -21.3,
 'TGAC': -22.7,
 'ACTG': -22.4,
 'AGTC': -21.0,
 'TCAG': -22.2,
 'CCGG': -19.9,
 # AC mis
 'AATC': 4.6,
 'ACTA': 14.6,
 'CAGC': 3.7,
 'CCGA': -0.6,
 'GACC': 14.2,
 'GCCA': -3.8,
 'TAAC': 8.0,
 'TCAA': 20.2,
 'CTAA': 4.6,
 'AACT': 20.2,
 'ATCA': 14.6,
 'CAAT': 8.0,
 'CGAC': 3.7,
 'ACCG': -3.8,
 'AGCC': -0.6,
 'CCAG': 14.2,
 # GA mis
 'AATG': -2.3,
 'AGTA': -2.3,
 'CAGG': -2.3,
 'CGGA': -13.2,
 'GACG': -1.0,
 'GGCA': 3.2,
 'TAAG': 0.7,
 'TGAA': 7.4,
 'GTAA': -2.3,
 'AAGT': 7.4,
 'ATGA': -2.3,
 'GAAT': 0.7,
 'GGAC': -2.3,
 'ACGG': 3.2,
 'AGGC': -13.2,
 'GCAG': -1.0,
 # CT mis
 'ACTT': 0.2,
 'ATTC': -6.2,
 'CCGT': -4.5,
 'CTGC': -6.1,
 'GCCT': 5.4,
 'GTCC': 13.5,
 'TCAT': 0.7,
 'TTAC': 0.7,
 'TTCA': 0.2,
 'CATT': 0.7,
 'CTTA': -6.2,
 'TACT': 0.7,
 'TGCC': -4.5,
 'CCTG': 13.5,
 'CGTC': -6.1,
 'TCCG': 5.4,
 # GT mis
 'AGTT': 0.9,
 'ATTG': -8.3,
 'CGGT': -11.7,
 'CTGG': -8.0,
 'GGCT': 10.4,
 'GGTT': 16.3,
 'GTCG': -12.3,
 'GTTG': 9.5,
 'TGAT': -1.7,
 'TGGT': -6.2,
 'TTAG': -5.3,
 'TTGA': 0.9,
 'GATT': -5.3,
 'GTTA': -8.3,
 'TAGT': -1.7,
 'TGGC': -11.7,
 'GCTG': -12.3,
 'GGTC': -8.0,
 'TCGG': 10.4,
 'TTGG': 16.3,
 # AA mis
 'AATA': 1.9,
 'CAGA': -4.3,
 'GACA': -9.9,
 'TAAA': 12.9,
 'ATAA': 1.9,
 'AAAT': 12.9,
 'AGAC': -4.3,
 'ACAG': -9.9,
 # CC mis
 'ACTC': -4.3,
 'CCGC': -7.1,
 'GCCC': 9.1,
 'TCAC': 16.3,
 'CTCA': -4.3,
 'CACT': 16.3,
 'CGCC': -7.1,
 'CCCG': 9.1,
 # GG mis
 'AGTG': -9.6,
 'CGGG': -15.4,
 'GGCG': -15.8,
 'TGAG': 3.7,
 'GTGA': -9.6,
 'GAGT': 3.7,
 'GGGC': -15.4,
 'GCGG': -15.8,
 # TT mis
 'ATTT': -10.9,
 'CTGT': -15.7,
 'GTCT': -8.5,
 'TTAT': -1.54,
 'TTTA': -10.9,
 'TATT': -1.54,
 'TGTC': -15.7,
 'TCTG': -8.5
}
dH_initiation = 0.2
dS_initiation = -5.7
dH_terminal_penalties = {
    "AT": 2.2,
    "TA": 2.2
}
dS_terminal_penalties = {
    "AT": 6.9,
    "TA": 6.9
}
