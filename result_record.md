| sick    |           |            |    | snli    |           |            |    | template                                                                                               |
|---------|-----------|------------|----|---------|-----------|------------|----|--------------------------------------------------------------------------------------------------------|
| roberta | roberta-l | bart-large | t5 | roberta | roberta-l | bart-large | t5 |                                                                                                         |          
| 25.45   | 20.61     | 23.64      |    | 32.50   | 34.40     | 32.10      |    | "if {} is 'true', the {} is '[MASK]'"                                                                   |                                                                   
| 29.09   | 18.99     | 18.79      |    | 32.50   | 34.60     | 26.30      |    | "Suppose that {} is: true, then {} is: [MASK]"                                                          |
| 26.46   | 30.71     | 26.87      |    | 34.30   | 40.80     | 32.50      |    | "When {} is true, then {} is [MASK]"                                                                    | 
| 30.30   | 16.77     | 17.98      |    | 33.10   |  38.90    | 37.10      |    | "When the premise '{}' is: true, then the hypothesis '{}' is: [MASK]"                                   | 
| 33.54   | 31.72     | 35.15      |    | 34.90   | 33.20   | 36.90      |    | "When the premise {} is true, then the hypothesis {} is: [MASK]"                                        |
| 29.70   | 24.04     | 28.89      |    | 29.50   | 34.70   | 32.50      |    | "Considering that the premise '{}' is 'true', then the hypothesis '{}' is: '[MASK]'"                    |
| 15.15   | 34.34     | 14.75      |    | 31.90   | 47.90   | 29.40      |    | "Knowing that the premise '{}' has a label 'true', then the hypothesis '{}' will have a label '[MASK]'" |
| 14.34   | 32.12     | 29.70      |    | 31.50   | 42.70   | 32.70      |    | "Regarding that the premise '{}' is: 'true', then the hypothesis '{}' will be '[MASK]'" |
| 33.94   | 23.43     | 23.64      |    | 33.40   | 36.70   | 32.40      |    | "{}: true, {}: [MASK]"    | 
| 35.76   | 30.30     | 33.74      |    | 34.80   | 37.30   | 31.00      |    | "{}? [MASK], {}"          | 
| 37.58   | 18.79     | 18.99      |    | 35.20   | 38.60   | 34.20      |    | "{}. [MASK], {}"          |
| 18.18   | 17.58     | 32.73      |    | 33.60   | 34.00   | 36.30      |    | "{}? [MASK] {}"           | 
| 35.96   | 28.69     | 57.17      |    | 35.60   | 39.60   | 33.10      |    | "{}. [MASK] , no , {}"    | 
| 56.36   | 36.36     | 57.37      |    | 32.40   | 37.30   | 35.00      |    | "{}. [MASK] this time {}" |