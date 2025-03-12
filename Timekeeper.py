# -*- coding: utf-8 -*-
"""
Created on Tue Mar 11 09:02:17 2025

@author: u6942852
"""

from datetime import datetime as dt
from datetime import timedelta as td
import pandas as pd

class Timekeeper:
    def __init__(self, path=None):
        self.path=path
        self.starttime=dt.now()
        
    def Update(self, name, time):
        if not hasattr(self, name):
            setattr(self, name, (td(0), 0))
        t, n = getattr(self, name)
        setattr(self, name, (t + time, n+1))
            
    def Print(self, console=True, file=None):
        names = [name for name in dir(self) if name[0] != '_' and name not in ( 'path', 'starttime', 'Update', 'Print')]
        results = pd.DataFrame({name:getattr(self, name) for name in names}, index=['time', 'calls']).T
        
        if file is None: 
            file = bool(self.path)
        if file is True:
            if self.path is None:
                raise Exception("Need a path to print to")
            else:
                results.to_csv(self.path, index=True, header=True)
        if console is True:
            print("Timekeeper","="*50, sep='\n')
            for row in results.iterrows():
                print(f'Function: {row[0]}. Calls: {row[1]["calls"]}. Time: {row[1]["time"]}.')


def keeptime(timekeeper, name):
    def decorator(func):
        def wrapper(*fargs):
            start=dt.now()
            ret=func(*fargs)
            timekeeper.Update(name, dt.now()-start)
            return ret
        return wrapper
    return decorator

if __name__=='__main__':

    from time import sleep
    from numba import njit
    tk=Timekeeper('test.csv')
    
    @keeptime(tk, 'func1')
    def func1(args=None):
        sleep(2)
        return 
    
    @keeptime(tk, 'func2')
    def func2(args=None):
        sleep(3)
        return
    
    @keeptime(tk, 'njit')
    @njit
    def func3(n=1_000_000):
        x = list(range(n))
        y = [y for y in x] 
        return 
    
    func1()
    func1()
    func2()
    func3(1)
    tk.Print()
    tk.path='test2.csv'
    func1()
    func2()
    func3()
    tk.Print()
    
    
        
        
        