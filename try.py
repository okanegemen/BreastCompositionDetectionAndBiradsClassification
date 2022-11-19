import pandas as pd

xls = pd.ExcelFile("/home/alican/Documents/yoloV5/INbreast Release 1.0/INbreast.xls")
sheetX = xls.parse(0)

print(sheetX[:][:410])