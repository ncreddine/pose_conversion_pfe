from datetime import time, timedelta


def subset(df, column, key) :
    return df[df[column] == key]



def get_time(isoformat):
    """ Creates a time object from isoformat
        @param  : str ISO format (hh:mm:ss.sss)
        ---------
        @return : time object
    """
    try :
        # split the microseconds with the rest
        s, microsecond = isoformat.split(".")
    except :
        # if failed then microseconds are not included and it equal to 0
        s = isoformat
        microsecond = 0
    # clip to max if greater than 999999
    microsecond = min( int(microsecond) , time.max.microsecond)
    hour, minute , second =  list(map(int, s.split(":")))
    return time(hour, minute, second, microsecond).isoformat()


def speakers() :
    return [ "almaram", "angelica", "chemistry", "conan", "ellen", "jon", "oliver", "rock", "seth", "shelly", "maher", "huckabee", "fallon", "lec_cosmic", 
             "colbert", "corden", "lec_evol", "minhaj", "bee", "lec_law", "ytch_dating", "lec_hist", "ytch_charisma", "ytch_prof", "ferguson", "noah" ]

