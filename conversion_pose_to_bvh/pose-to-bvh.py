# Nous utilisons ce code pour convertir les poses que nous avons trouvé (sous forme de numpy) aux fichiers .bvh. 
# Nous avons ecrit ce code en nous inspirant du github suivant:  https://github.com/Michele1996/PFE-OpenPose-to-VAE-to-BVH

import numpy

bvh_joints="""HIERARCHY
ROOT hip
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation
  JOINT abdomen
  {
    OFFSET 0 20.6881 -0.73152
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT chest
    {
      OFFSET 0 11.7043 -0.48768
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT neck
      {
          OFFSET 0 22.1894 -2.19456
          CHANNELS 3 Zrotation Xrotation Yrotation 
          JOINT neck1
          {
            OFFSET 0.000000 5.364170 1.574630
            CHANNELS 3 Zrotation Xrotation Yrotation 
            JOINT head
            {
              OFFSET 0.000000 5.364141 1.574630
              CHANNELS 3 Zrotation Xrotation Yrotation 
              JOINT __jaw
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT jaw
                {
                  OFFSET 0.000000 -13.499860 2.500710
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT special04
                  {
                    OFFSET -0.000000 -6.835370 4.375500
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT oris02
                    {
                      OFFSET 0.000000 1.711150 2.820850
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT oris01
                      {
                        OFFSET -0.000000 0.972390 0.845650
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        End Site
                        {
                          OFFSET 0.000000 1.162291 0.607091
                        }
                      }
                    }
                    JOINT oris06.l
                    {
                      OFFSET 0.000000 1.711150 2.820850
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT oris07.l
                      {
                        OFFSET 1.168850 0.445180 0.506110
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        End Site
                        {
                          OFFSET 0.450611 1.195178 0.204519
                        }
                      }
                    }
                    JOINT oris06.r
                    {
                      OFFSET 0.000000 1.711150 2.820850
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT oris07.r
                      {
                        OFFSET -1.168850 0.445180 0.506110
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        End Site
                        {
                          OFFSET -0.450611 1.195173 0.204519
                        }
                      }
                    }
                  }
                  JOINT tongue00
                  {
                    OFFSET -0.000000 -6.835370 4.375500
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT tongue01
                    {
                      OFFSET 0.000000 3.973650 -3.762340
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT tongue02
                      {
                        OFFSET 0.000000 0.429760 2.924710
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT tongue03
                        {
                          OFFSET 0.000000 0.018530 2.059010
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          JOINT __tongue04
                          {
                            OFFSET 0.000000 -0.440240 0.838860
                            CHANNELS 3 Zrotation Xrotation Yrotation 
                            JOINT tongue04
                            {
                              OFFSET 0.000000 0.000000 0.000000
                              CHANNELS 3 Zrotation Xrotation Yrotation 
                              End Site
                              {
                                OFFSET 0.000000 -0.440230 0.838860
                              }
                            }
                          }
                          JOINT tongue07.l
                          {
                            OFFSET 0.000000 -0.440240 0.838860
                            CHANNELS 3 Zrotation Xrotation Yrotation 
                            End Site
                            {
                              OFFSET 1.160923 -0.331531 0.018227
                            }
                          }
                          JOINT tongue07.r
                          {
                            OFFSET 0.000000 -0.440240 0.838860
                            CHANNELS 3 Zrotation Xrotation Yrotation 
                            End Site
                            {
                              OFFSET -1.160922 -0.331531 0.018227
                            }
                          }
                        }
                        JOINT tongue06.l
                        {
                          OFFSET 0.000000 0.018530 2.059010
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          End Site
                          {
                            OFFSET 1.644752 -0.526075 -0.203281
                          }
                        }
                        JOINT tongue06.r
                        {
                          OFFSET 0.000000 0.018530 2.059010
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          End Site
                          {
                            OFFSET -1.644752 -0.526075 -0.203282
                          }
                        }
                      }
                      JOINT tongue05.l
                      {
                        OFFSET 0.000000 0.429760 2.924710
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        End Site
                        {
                          OFFSET 1.971028 -0.388618 0.239206
                        }
                      }
                      JOINT tongue05.r
                      {
                        OFFSET 0.000000 0.429760 2.924710
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        End Site
                        {
                          OFFSET -1.971028 -0.388618 0.239205
                        }
                      }
                    }
                  }
                }
              }
              JOINT __levator02.l
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT levator02.l
                {
                  OFFSET 0.313580 -11.321120 11.599360
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT levator03.l
                  {
                    OFFSET 1.681690 -1.563730 -1.357570
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT levator04.l
                    {
                      OFFSET 0.504730 -1.676760 -0.058160
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT levator05.l
                      {
                        OFFSET 0.145440 -1.643170 -0.225470
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        End Site
                        {
                          OFFSET -0.310116 -0.760198 -0.121474
                        }
                      }
                    }
                  }
                }
              }
              JOINT __levator02.r
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT levator02.r
                {
                  OFFSET -0.313580 -11.321120 11.599360
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT levator03.r
                  {
                    OFFSET -1.681690 -1.563740 -1.357570
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT levator04.r
                    {
                      OFFSET -0.504730 -1.676750 -0.058160
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT levator05.r
                      {
                        OFFSET -0.145440 -1.643170 -0.225470
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        End Site
                        {
                          OFFSET 0.310116 -0.760193 -0.121474
                        }
                      }
                    }
                  }
                }
              }
              JOINT __special01
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT special01
                {
                  OFFSET -0.000000 -14.026930 -5.716970
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT oris04.l
                  {
                    OFFSET -0.000000 -0.492640 17.312620
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT oris03.l
                    {
                      OFFSET 1.215520 -0.627430 -0.393050
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      End Site
                      {
                        OFFSET 0.288776 -0.560899 -0.149645
                      }
                    }
                  }
                  JOINT oris04.r
                  {
                    OFFSET -0.000000 -0.492640 17.312620
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT oris03.r
                    {
                      OFFSET -1.215520 -0.627440 -0.393050
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      End Site
                      {
                        OFFSET -0.288776 -0.560894 -0.149645
                      }
                    }
                  }
                  JOINT oris06
                  {
                    OFFSET -0.000000 -0.492640 17.312620
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT oris05
                    {
                      OFFSET -0.000000 -0.486000 0.000950
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      End Site
                      {
                        OFFSET 0.000000 -0.630493 0.197635
                      }
                    }
                  }
                }
              }
              JOINT __special03
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT special03
                {
                  OFFSET 0.000000 -13.499860 2.500710
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT __levator06.l
                  {
                    OFFSET -0.000000 1.035800 10.090229
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT levator06.l
                    {
                      OFFSET 0.522240 -0.615720 0.045900
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      End Site
                      {
                        OFFSET 1.107505 -0.131898 -2.142227
                      }
                    }
                  }
                  JOINT __levator06.r
                  {
                    OFFSET -0.000000 1.035800 10.090229
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT levator06.r
                    {
                      OFFSET -0.522240 -0.615730 0.045900
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      End Site
                      {
                        OFFSET -1.107506 -0.131893 -2.142227
                      }
                    }
                  }
                }
              }
              JOINT special06.l
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT special05.l
                {
                  OFFSET 2.108890 0.153870 5.595070
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT eye.l
                  {
                    OFFSET 0.857170 -10.254801 2.414670
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    End Site
                    {
                      OFFSET 0.111609 0.006490 4.051424
                    }
                  }
                  JOINT orbicularis03.l
                  {
                    OFFSET 0.857170 -10.254801 2.414670
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    End Site
                    {
                      OFFSET 0.097789 1.146431 3.788029
                    }
                  }
                  JOINT orbicularis04.l
                  {
                    OFFSET 0.857170 -10.254801 2.414670
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    End Site
                    {
                      OFFSET 0.113609 -1.130505 3.863064
                    }
                  }
                }
              }
              JOINT special06.r
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT special05.r
                {
                  OFFSET -2.108890 0.153870 5.595070
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT eye.r
                  {
                    OFFSET -0.857170 -10.254801 2.414670
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    End Site
                    {
                      OFFSET -0.111609 0.006490 4.051424
                    }
                  }
                  JOINT orbicularis03.r
                  {
                    OFFSET -0.857170 -10.254801 2.414670
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    End Site
                    {
                      OFFSET -0.097789 1.146431 3.788029
                    }
                  }
                  JOINT orbicularis04.r
                  {
                    OFFSET -0.857170 -10.254801 2.414670
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    End Site
                    {
                      OFFSET -0.113609 -1.130505 3.863064
                    }
                  }
                }
              }
              JOINT __temporalis01.l
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT temporalis01.l
                {
                  OFFSET 6.332510 -9.444281 6.595120
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT oculi02.l
                  {
                    OFFSET -0.804920 0.053010 1.621140
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT oculi01.l
                    {
                      OFFSET -2.161570 1.690970 2.142660
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      End Site
                      {
                        OFFSET -1.740622 0.281157 0.647323
                      }
                    }
                  }
                }
              }
              JOINT __temporalis01.r
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT temporalis01.r
                {
                  OFFSET -6.332510 -9.444281 6.595120
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT oculi02.r
                  {
                    OFFSET 0.804920 0.053010 1.621140
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT oculi01.r
                    {
                      OFFSET 2.161570 1.690970 2.142660
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      End Site
                      {
                        OFFSET 1.740622 0.281157 0.647323
                      }
                    }
                  }
                }
              }
              JOINT __temporalis02.l
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT temporalis02.l
                {
                  OFFSET 6.377600 -11.680510 6.235180
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT risorius02.l
                  {
                    OFFSET -0.814250 0.451130 1.721730
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT risorius03.l
                    {
                      OFFSET -0.649710 -2.514660 0.612550
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      End Site
                      {
                        OFFSET 0.479556 -1.760402 -1.642659
                      }
                    }
                  }
                }
              }
              JOINT __temporalis02.r
              {
                OFFSET 0.000000 13.604700 -0.502080
                CHANNELS 3 Zrotation Xrotation Yrotation 
                JOINT temporalis02.r
                {
                  OFFSET -6.377600 -11.680510 6.235180
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT risorius02.r
                  {
                    OFFSET 0.814250 0.451130 1.721730
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT risorius03.r
                    {
                      OFFSET 0.649710 -2.514660 0.612550
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      End Site
                      {
                        OFFSET -0.479556 -1.760402 -1.642659
                      }
                    }
                  }
                }
              }
            }
          }
        }
      JOINT rCollar
      {
        OFFSET -2.68224 19.2634 -4.8768
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT rShldr
        {
          OFFSET -8.77824 -1.95073 1.46304
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT rForeArm
          {
            OFFSET -28.1742 -1.7115 0.48768
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT rHand
            {
                  OFFSET -21.049400 0.002190 -0.634230
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT metacarpal1.r
                  {
                    OFFSET -2.815680 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT finger2-1.r
                    {
                      OFFSET -6.292930 0.272380 2.520090
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger2-2.r
                      {
                        OFFSET -2.310530 -0.320530 -0.060510
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger2-3.r
                        {
                          OFFSET -2.051030 -0.295400 -0.164890
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          End Site
                          {
                            OFFSET -2.376838 -0.681367 -0.183877
                          }
                        }
                      }
                    }
                  }
                  JOINT metacarpal2.r
                  {
                    OFFSET -2.815680 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT finger3-1.r
                    {
                      OFFSET -6.313640 0.626130 0.318530
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger3-2.r
                      {
                        OFFSET -3.015730 -0.589480 -0.088540
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger3-3.r
                        {
                          OFFSET -2.482120 -0.426280 0.076670
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          End Site
                          {
                            OFFSET -2.344174 -0.731969 0.003260
                          }
                        }
                      }
                    }
                  }
                  JOINT __metacarpal3.r
                  {
                    OFFSET -2.815680 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT metacarpal3.r
                    {
                      OFFSET -0.606080 -0.162120 -1.874870
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger4-1.r
                      {
                        OFFSET -5.355730 0.702040 0.402510
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger4-2.r
                        {
                          OFFSET -2.643900 -0.485530 -0.117520
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          JOINT finger4-3.r
                          {
                            OFFSET -2.215850 -0.353160 0.066220
                            CHANNELS 3 Zrotation Xrotation Yrotation 
                            End Site
                            {
                              OFFSET -2.350273 -0.621223 -0.046375
                            }
                          }
                        }
                      }
                    }
                  }
                  JOINT __metacarpal4.r
                  {
                    OFFSET -2.815680 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT metacarpal4.r
                    {
                      OFFSET -0.606080 -0.162120 -1.874870
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger5-1.r
                      {
                        OFFSET -4.761700 0.175470 -1.109590
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger5-2.r
                        {
                          OFFSET -1.916360 -0.173360 -0.146170
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          JOINT finger5-3.r
                          {
                            OFFSET -1.411290 -0.108670 -0.020110
                            CHANNELS 3 Zrotation Xrotation Yrotation 
                            End Site
                            {
                              OFFSET -1.799226 -0.102363 -0.078601
                            }
                          }
                        }
                      }
                    }
                  }
                  JOINT __rthumb
                  {
                    OFFSET -2.815680 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT rthumb
                    {
                      OFFSET -0.283040 -0.142720 1.950690
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger1-2.r
                      {
                        OFFSET -0.915590 -2.152150 1.546760
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger1-3.r
                        {
                          OFFSET -3.213140 -0.470060 0.247480
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          End Site
                          {
                            OFFSET -2.521224 -0.161543 -0.511272
                          }
                        }
                      }
                    }
                  }
            }
          }
        }
      }
      JOINT lCollar
      {
        OFFSET 2.68224 19.2634 -4.8768
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT lShldr
        {
          OFFSET 8.77824 -1.95073 1.46304
          CHANNELS 3 Zrotation Xrotation Yrotation
          JOINT lForeArm
          {
            OFFSET 28.1742 -1.7115 0.48768
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT lHand
            {
             
                  OFFSET 21.049408 0.002200 -0.634230
                  CHANNELS 3 Zrotation Xrotation Yrotation 
                  JOINT metacarpal1.l
                  {
                    OFFSET 2.815670 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT finger2-1.l
                    {
                      OFFSET 6.292930 0.272390 2.520090
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger2-2.l
                      {
                        OFFSET 2.310530 -0.320520 -0.060510
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger2-3.l
                        {
                          OFFSET 2.051030 -0.295400 -0.164880
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          End Site
                          {
                            OFFSET 2.376823 -0.681367 -0.183876
                          }
                        }
                      }
                    }
                  }
                  JOINT metacarpal2.l
                  {
                    OFFSET 2.815670 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT finger3-1.l
                    {
                      OFFSET 6.313640 0.626120 0.318530
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger3-2.l
                      {
                        OFFSET 3.015730 -0.589470 -0.088540
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger3-3.l
                        {
                          OFFSET 2.482120 -0.426270 0.076670
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          End Site
                          {
                            OFFSET 2.344170 -0.731978 0.003260
                          }
                        }
                      }
                    }
                  }
                  JOINT __metacarpal3.l
                  {
                    OFFSET 2.815670 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT metacarpal3.l
                    {
                      OFFSET 0.606080 -0.162120 -1.874870
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger4-1.l
                      {
                        OFFSET 5.355730 0.702050 0.402510
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger4-2.l
                        {
                          OFFSET 2.643900 -0.485530 -0.117510
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          JOINT finger4-3.l
                          {
                            OFFSET 2.215840 -0.353150 0.066210
                            CHANNELS 3 Zrotation Xrotation Yrotation 
                            End Site
                            {
                              OFFSET 2.350273 -0.621228 -0.046377
                            }
                          }
                        }
                      }
                    }
                  }
                  JOINT __metacarpal4.l
                  {
                    OFFSET 2.815670 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT metacarpal4.l
                    {
                      OFFSET 0.606080 -0.162120 -1.874870
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger5-1.l
                      {
                        OFFSET 4.761700 0.175480 -1.109600
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger5-2.l
                        {
                          OFFSET 1.916350 -0.173360 -0.146170
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          JOINT finger5-3.l
                          {
                            OFFSET 1.411290 -0.108670 -0.020110
                            CHANNELS 3 Zrotation Xrotation Yrotation 
                            End Site
                            {
                              OFFSET 1.799216 -0.102372 -0.078600
                            }
                          }
                        }
                      }
                    }
                  }
                  JOINT __lthumb
                  {
                    OFFSET 2.815670 -0.279180 0.531660
                    CHANNELS 3 Zrotation Xrotation Yrotation 
                    JOINT lthumb
                    {
                      OFFSET 0.283040 -0.142710 1.950690
                      CHANNELS 3 Zrotation Xrotation Yrotation 
                      JOINT finger1-2.l
                      {
                        OFFSET 0.915930 -2.151960 1.546820
                        CHANNELS 3 Zrotation Xrotation Yrotation 
                        JOINT finger1-3.l
                        {
                          OFFSET 3.213210 -0.469680 0.247300
                          CHANNELS 3 Zrotation Xrotation Yrotation 
                          End Site
                          {
                            OFFSET 2.521210 -0.161290 -0.511422
                          }
                        }
                      }
                    }
                  }
            }
          }
        }
      }
    }
  }
  JOINT rButtock
  {
    OFFSET -8.77824 4.35084 1.2192
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT rThigh
    {
      OFFSET 0 -1.70687 -2.19456
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT rShin
      {
        OFFSET 0 -36.8199 0.73152
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT rFoot
        {
         
       OFFSET 0.73152 -45.1104 -5.12064
       CHANNELS 3 Zrotation Xrotation Yrotation
       JOINT toe1-1.Right
       {
        OFFSET 2.454000 -4.050002 13.194999
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe1-2.Right
        {
         OFFSET -0.214000 -0.646000 2.427000
         CHANNELS 3 Zrotation Xrotation Yrotation
         End Site
         {
          OFFSET -0.401900 -0.827789 2.725930
         }
        }
       }
       JOINT toe2-1.Right
       {
        OFFSET 0.177000 -4.299998 13.329000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe2-2.Right
        {
         OFFSET -0.177000 -0.323000 2.039000
         CHANNELS 3 Zrotation Xrotation Yrotation
         JOINT toe2-3.Right
         {
          OFFSET -0.067000 -0.440998 1.248000
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
           OFFSET -0.042990 -0.647306 1.660872
          }
         }
        }
       }
       JOINT toe3-1.Right
       {
        OFFSET -1.396000 -4.461999 13.078999
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe3-2.Right
        {
         OFFSET -0.161000 -0.247002 1.809000
         CHANNELS 3 Zrotation Xrotation Yrotation
         JOINT toe3-3.Right
         {
          OFFSET -0.033000 -0.441999 1.202000
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
           OFFSET 0.032040 -0.433550 1.271800
          }
         }
        }
       }
       JOINT toe4-1.Right
       {
        OFFSET -2.888001 -4.480000 12.376999
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe4-2.Right
        {
         OFFSET -0.160000 -0.331998 1.491001
         CHANNELS 3 Zrotation Xrotation Yrotation
         JOINT toe4-3.Right
         {
          OFFSET 0.035999 -0.251002 1.138999
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
           OFFSET -0.088911 -0.568814 0.969530
          }
         }
        }
       }
       JOINT toe5-1.Right
       {
        OFFSET -4.257999 -4.467001 11.711999
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe5-2.Right
        {
         OFFSET -0.046000 -0.265999 0.982000
         CHANNELS 3 Zrotation Xrotation Yrotation
         JOINT toe5-3.Right
         {
          OFFSET 0.086999 -0.372000 0.791000
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
           OFFSET -0.044329 -0.555482 1.085780
          }
         }
        }
       }
        }
      }
    }
  }
  JOINT lButtock
  {
    OFFSET 8.77824 4.35084 1.2192
    CHANNELS 3 Zrotation Xrotation Yrotation
    JOINT lThigh
    {
      OFFSET 0 -1.70687 -2.19456
      CHANNELS 3 Zrotation Xrotation Yrotation
      JOINT lShin
      {
        OFFSET 0 -36.8199 0.73152
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT lFoot
        { 

       OFFSET -0.73152 -45.1104 -5.12064
       CHANNELS 3 Zrotation Xrotation Yrotation
       JOINT toe1-1.Left
       {
        OFFSET -2.454000 -4.050002 13.194999
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe1-2.Left
        {
         OFFSET 0.214000 -0.646000 2.427000
         CHANNELS 3 Zrotation Xrotation Yrotation
         End Site
         {
          OFFSET 0.401900 -0.827789 2.725930
         }
        }
       }
       JOINT toe2-1.Left
       {
        OFFSET -0.177000 -4.299998 13.329000
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe2-2.Left
        {
         OFFSET 0.177000 -0.323000 2.039000
         CHANNELS 3 Zrotation Xrotation Yrotation
         JOINT toe2-3.Left
         {
          OFFSET 0.067000 -0.440998 1.248000
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
           OFFSET 0.042990 -0.647306 1.660872
          }
         }
        }
       }
       JOINT toe3-1.Left
       {
        OFFSET 1.396000 -4.461999 13.078999
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe3-2.Left
        {
         OFFSET 0.161000 -0.247002 1.809000
         CHANNELS 3 Zrotation Xrotation Yrotation
         JOINT toe3-3.Left
         {
          OFFSET 0.033000 -0.441999 1.202000
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
           OFFSET -0.032040 -0.433550 1.271800
          }
         }
        }
       }
       JOINT toe4-1.Left
       {
        OFFSET 2.888001 -4.480000 12.376999
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe4-2.Left
        {
         OFFSET 0.160000 -0.331998 1.491001
         CHANNELS 3 Zrotation Xrotation Yrotation
         JOINT toe4-3.Left
         {
          OFFSET -0.035999 -0.251002 1.138999
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
           OFFSET 0.088911 -0.568814 0.969530
          }
         }
        }
       }
       JOINT toe5-1.Left
       {
        OFFSET 4.257999 -4.467001 11.711999
        CHANNELS 3 Zrotation Xrotation Yrotation
        JOINT toe5-2.Left
        {
         OFFSET 0.046000 -0.265999 0.982000
         CHANNELS 3 Zrotation Xrotation Yrotation
         JOINT toe5-3.Left
         {
          OFFSET -0.086999 -0.372000 0.791000
          CHANNELS 3 Zrotation Xrotation Yrotation
          End Site
          {
           OFFSET 0.044329 -0.555482 1.085780
          }
         }
        }
       }
        }
      }
    }
  }
}

MOTION
"""
#les articulations que j'ai enlevé : Nose, LHandRooth, LHandThumb4,RHandRooth,RHandThumb4
print(bvh_joints)
dictionary_skeleton={"neck":0,"rShldr":1,"rForeArm":2,"rHand":3,"lShldr":4,"lForeArm":5,"lHand":6,"eye.r":7,"eye.l":8,"lthumb":9,"finger1-2.l":10,"finger1-3.l":11,"metacarpal4.l":12,"finger5-1.l":13,"finger5-2.l":14,"finger5-3.l":15,"metacarpal3.l":16,"finger4-1.l":17,"finger4-2.l":18,"finger4-3.l":19,"metacarpal2.l":20,"finger3-1.l":21,"finger3-2.l":22,"finger3-3.l":23,"metacarpal1.l":24,"finger2-1.l":25,"finger2-2.l":26,"finger2-3.l":27,"rthumb":28,"finger1-2.r":29,"finger1-3.r":30,"metacarpal4.r":31,"finger5-1.r":32,"finger5-2.r":33,"finger5-3.r":34,"metacarpal3.r":35,"finger4-1.r":36,"finger4-2.r":37,"finger4-3.r":38,"metacarpal2.r":39,"finger3-1.r":40,"finger3-2.r":41,"finger3-3.r":42,"metacarpal1.r":43,"finger2-1.r":44,"finger2-2.r":45,"finger2-3.r":46}
dictionary_skeleton_2={"neck":0,"rShldr":1,"rForeArm":2,"rHand":3,"lShldr":4,"lForeArm":5,"lHand":6,"eye.r":7,"eye.l":8,"lthumb":9,"finger1-2.l":10,"finger1-3.l":11,"metacarpal4.l":12,"finger5-1.l":13,"finger5-2.l":14,"finger5-3.l":15,"metacarpal3.l":16,"finger4-1.l":17,"finger4-2.l":18,"finger4-3.l":19,"metacarpal2.l":20,"finger3-1.l":21,"finger3-2.l":22,"finger3-3.l":23,"metacarpal1.l":24,"finger2-1.l":25,"finger2-2.l":26,"finger2-3.l":27,"rthumb":28,"finger1-2.r":29,"finger1-3.r":30,"metacarpal4.r":31,"finger5-1.r":32,"finger5-2.r":33,"finger5-3.r":34,"metacarpal3.r":35,"finger4-1.r":36,"finger4-2.r":37,"finger4-3.r":38,"metacarpal2.r":39,"finger3-1.r":40,"finger3-2.r":41,"finger3-3.r":42,"metacarpal1.r":43,"finger2-1.r":44,"finger2-2.r":45,"finger2-3.r":46}
dictionary_skeleton_3={"neck":0,"rShldr":1,"rForeArm":2,"rHand":3,"lShldr":4,"lForeArm":5,"lHand":6,"eye.r":7,"eye.l":8,"lthumb":9,"finger1-2.l":10,"finger1-3.l":11,"metacarpal4.l":12,"finger5-1.l":13,"finger5-2.l":14,"finger5-3.l":15,"metacarpal3.l":16,"finger4-1.l":17,"finger4-2.l":18,"finger4-3.l":19,"metacarpal2.l":20,"finger3-1.l":21,"finger3-2.l":22,"finger3-3.l":23,"metacarpal1.l":24,"finger2-1.l":25,"finger2-2.l":26,"finger2-3.l":27,"rthumb":28,"finger1-2.r":29,"finger1-3.r":30,"metacarpal4.r":31,"finger5-1.r":32,"finger5-2.r":33,"finger5-3.r":34,"metacarpal3.r":35,"finger4-1.r":36,"finger4-2.r":37,"finger4-3.r":38,"metacarpal2.r":39,"finger3-1.r":40,"finger3-2.r":41,"finger3-3.r":42,"metacarpal1.r":43,"finger2-1.r":44,"finger2-2.r":45,"finger2-3.r":46}
x=numpy.load("pred.npy",allow_pickle=True)


pred_list=numpy.zeros(((len(x),len(x[0])//3,3)))
new_list=numpy.zeros(((len(x),len(x[0])//3-5,3)))

for frame in range(len(x)):
    for joint in range(len(x[0])//3):
        pred_list[frame,joint,0], pred_list[frame,joint,1],pred_list[frame,joint,2]=x[frame,joint], x[frame,joint+1],x[frame,joint+2]

for frame in range(len(x)):
    for joint in range(len(x[0])//3-5):
        if joint==7 or joint==10 or joint==14 or joint==31 or joint==35:
            pass
        else:
            new_list[frame,joint]=pred_list[frame,joint]

x= new_list

diction={i : x[i] for i in  range(len(x))}
#print(dictionary_skeleton)
bvh = open("test_bvh", "w")
bvh.write(bvh_joints)
bvh.close()
new_bvh=open("new.bvh","w")
bvh_2=open("test_bvh", "r")
p_bvh=bvh_2.readlines()
lines=[]
for line in p_bvh:
    if("JOINT" in line or "ROOT" in line):
        lines.append(line.replace(" ","").replace("JOINT","").replace("ROOT","").replace("\t",""))
dictionary_bvh={i:lines[i].replace("\n","") for i in range(len(lines))}
#for line in p_bvh:
#    new_bvh.write(line)
#    if("MOTION" in line):
#        break
new_bvh.write(bvh_joints)
new_bvh.write("Frames: "+str(len(x))+"\n")
new_bvh.write("Frame Time: 0.04\n")
new_line=[]
index=0
num_val=3
correct_skel_added=0
for i in range(len(x)):
    index=0
    print("PROCESSING "+str(i+1)+"/"+str(len(x))+" FRAME")
    for p in range(len(lines)):

        if(dictionary_bvh[p] in dictionary_skeleton_3.keys()):
           correct_skel_added+=1
           val=diction[i][dictionary_skeleton_3[dictionary_bvh[p]]]
           print(val)
           new_line.append(val[0])
           new_line.append(val[1])
           new_line.append(val[2])
#if you have also the angles for the hips you can add it here
           if(dictionary_bvh[p]=="hip"):
              new_line.append(0)
              new_line.append(0)
              new_line.append(0)
        else:
            for k in range(3):
                new_line.append(0)
    string=""
    for z in new_line:
        string=string+str(z)+" ";
    string=string+"\n"
    string=string.replace("[","").replace("]","")
    print(len(string.split(" ")), correct_skel_added)
    new_bvh.write(string)
    new_line=[]
    correct_skel_added=0