import math
import Weibull
import statistics
import numpy as np
import sympy as sp
import pandas as pd
import xarray as xr
import requests_cache
from py_wake import NOJ
import openmeteo_requests
from py_wake.site import XRSite
import matplotlib.pyplot as plt
from retry_requests import retry
from windrose import WindroseAxes
from scipy.integrate import trapezoid
from py_wake import BastankhahGaussian
from py_wake.examples.data.dtu10mw import DTU10MW
from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular

# Weather report import from Open-Meteo
cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)
url = "https://archive-api.open-meteo.com/v1/archive"

# Get user input for latitude and longitude
latitude = float(input("Enter latitude for desired position: "))
longitude = float(input("Enter longitude for desired position: "))
params = {
    "latitude": latitude,
    "longitude": longitude,
    "start_date": "2014-01-01",
    "end_date": "2024-01-01",
    "hourly": ["wind_speed_10m", "wind_speed_100m", "wind_direction_100m"],
    "wind_speed_unit": "ms",
}
responses = openmeteo.weather_api(url, params=params)

# Process first location. Add a for-loop for multiple locations or weather models
response = responses[0]
num_rows = int(input("Enter the number of turbines per column: "))
num_cols = int(input("Enter the number of turbines per row: "))
turbine_spacing_x = int(input("Enter the spacing between turbines in meters on the x-axis: "))
turbine_spacing_y = int(input("Enter the spacing between turbines in meters on the y-axis: "))
print("Choose the layout type:")
print("1. Back-to-back layout")
print("2. Zigzag layout")
menu = int(input("Enter your choice (1 or 2): "))
print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
# print(f"Elevation {response.Elevation()} m asl")

# Import from open-meteo
hourly = response.Hourly()
hourly_wind_speed_10m = hourly.Variables(0).ValuesAsNumpy()
hourly_wind_speed_100m = hourly.Variables(1).ValuesAsNumpy()
hourly_wind_direction_100m = hourly.Variables(2).ValuesAsNumpy()

# Wind speed at 150m calculations
hourly_wind_speed_150m = hourly_wind_speed_100m * ((math.log(150 / 0.05)) / (math.log(100 / 0.05)))
hourly_wind_speed_150m_mean = statistics.mean(hourly_wind_speed_150m)
print('Wind mean:', hourly_wind_speed_150m_mean)

# Standard deviation calculations
hourly_wind_deviation = hourly_wind_speed_150m - hourly_wind_speed_150m_mean
hourly_standard_deviation = (math.sqrt((sum(hourly_wind_deviation ** 2)) / len(hourly_wind_speed_150m)))
print('Standard deviation:', hourly_standard_deviation)

# Weibull shape and scale factor calculations
weibull_k = ((hourly_standard_deviation / hourly_wind_speed_150m_mean) ** (-1.090))  # Pre-defined formula
print('weibull k:', weibull_k)
weibull_c = ((2 * hourly_wind_speed_150m_mean) / (math.sqrt(math.pi)))  # Pre-defined formula
print('weibull c:', weibull_c)

# Creating a dictionary containing hourly data
hourly_data = {"date": pd.date_range(
    start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
    end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
    freq=pd.Timedelta(seconds=hourly.Interval()),
    inclusive="left"
), "wind_speed_150m": hourly_wind_speed_150m, "wind_direction_100m": hourly_wind_direction_100m}

# Creating a dataframe with the corresponding hourly data
hourly_dataframe = pd.DataFrame(data=hourly_data)

# Calculating average wind direction
avg_wd = np.mean(hourly_wind_direction_100m)

# Initialization of Twelve Wind Direction Lists for Data Categorization
list_30 = []  # Represents wind directions ranging from 0 degrees to 30 degrees.
list_60 = []  # Represents wind directions ranging from 31 degrees to 60 degrees.
list_90 = []  # Represents wind directions ranging from 61 degrees to 90 degrees.
list_120 = []  # Represents wind directions ranging from 91 degrees to 120 degrees.
list_150 = []  # Represents wind directions ranging from 121 degrees to 150 degrees.
list_180 = []  # Represents wind directions ranging from 151 degrees to 180 degrees.
list_210 = []  # Represents wind directions ranging from 181 degrees to 210 degrees.
list_240 = []  # Represents wind directions ranging from 211 degrees to 240 degrees.
list_270 = []  # Represents wind directions ranging from 241 degrees to 270 degrees.
list_300 = []  # Represents wind directions ranging from 271 degrees to 300 degrees.
list_330 = []  # Represents wind directions ranging from 301 degrees to 330 degrees.
list_360 = []  # Represents wind directions ranging from 331 degrees to 360 degrees.
# print(hourly_dataframe)

# Wind Speed Data Organization by Directional Categories
for i in range(len(hourly_dataframe['wind_direction_100m'])):
    if 0 <= hourly_dataframe['wind_direction_100m'].iloc[i] <= 30:
        list_30.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 30 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 60:
        list_60.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 60 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 90:
        list_90.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 90 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 120:
        list_120.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 120 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 150:
        list_150.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 150 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 180:
        list_180.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 180 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 210:
        list_210.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 210 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 240:
        list_240.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 240 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 270:
        list_270.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 270 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 300:
        list_300.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 300 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 330:
        list_330.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    elif 330 < hourly_dataframe['wind_direction_100m'].iloc[i] <= 360:
        list_360.append(hourly_dataframe['wind_speed_150m'].iloc[i])
    else:
        print(hourly_dataframe['wind_speed_150m'].iloc[i])
    # if i == 1000:
    #     break

div = len(hourly_dataframe)

# Calculating the average wind speed in each wind direction
avg_30 = np.mean(list_30)
avg_60 = np.mean(list_60)
avg_90 = np.mean(list_90)
avg_120 = np.mean(list_120)
avg_150 = np.mean(list_150)
avg_180 = np.mean(list_180)
avg_210 = np.mean(list_210)
avg_240 = np.mean(list_240)
avg_270 = np.mean(list_270)
avg_300 = np.mean(list_300)
avg_330 = np.mean(list_330)
avg_360 = np.mean(list_360)

# Calculating average wind direction for each category
dir_30 = round(len(list_30) / div, 3)
dir_60 = round(len(list_60) / div, 3)
dir_90 = round(len(list_90) / div, 3)
dir_120 = round(len(list_120) / div, 3)
dir_150 = round(len(list_150) / div, 3)
dir_180 = round(len(list_180) / div, 3)
dir_210 = round(len(list_210) / div, 3)
dir_240 = round(len(list_240) / div, 3)
dir_270 = round(len(list_270) / div, 3)
dir_300 = round(len(list_300) / div, 3)
dir_330 = round(len(list_330) / div, 3)
dir_360 = round(len(list_360) / div, 3)

# Sorting the average direction of each category into a list
f = []
f.append(dir_30)
f.append(dir_60)
f.append(dir_90)
f.append(dir_120)
f.append(dir_150)
f.append(dir_180)
f.append(dir_210)
f.append(dir_240)
f.append(dir_270)
f.append(dir_300)
f.append(dir_330)
f.append(dir_360)


# General standard deviation calculation
def calc(list_x, avg_x):
    for i in range(len(list_x)):
        list_x[i] = list_x[i] - avg_x
        list_x[i] = list_x[i] ** 2
    s = np.sum(list_x) / len(list_x)
    sd = math.sqrt(s)
    return sd


# Calculating standard deviation for each direction
sd30 = calc(list_30, avg_30)
sd60 = calc(list_60, avg_60)
sd90 = calc(list_90, avg_90)
sd120 = calc(list_120, avg_120)
sd150 = calc(list_150, avg_150)
sd180 = calc(list_180, avg_180)
sd210 = calc(list_210, avg_210)
sd240 = calc(list_240, avg_240)
sd270 = calc(list_270, avg_270)
sd300 = calc(list_300, avg_300)
sd330 = calc(list_330, avg_330)
sd360 = calc(list_360, avg_360)

# Calculating the Weibull shape parameter, k, for each direction
k_30 = ((sd30 / avg_30) ** (-1.090))
k_60 = ((sd60 / avg_60) ** (-1.090))
k_90 = ((sd90 / avg_90) ** (-1.090))
k_120 = ((sd120 / avg_120) ** (-1.090))
k_150 = ((sd150 / avg_150) ** (-1.090))
k_180 = ((sd180 / avg_180) ** (-1.090))
k_210 = ((sd210 / avg_210) ** (-1.090))
k_240 = ((sd240 / avg_240) ** (-1.090))
k_270 = ((sd270 / avg_270) ** (-1.090))
k_300 = ((sd300 / avg_300) ** (-1.090))
k_330 = ((sd330 / avg_330) ** (-1.090))
k_360 = ((sd360 / avg_360) ** (-1.090))

# Calculating the Weibull scale parameter, c, for each direction
c_30 = ((2 * avg_30) / (math.sqrt(math.pi)))
c_60 = ((2 * avg_60) / (math.sqrt(math.pi)))
c_90 = ((2 * avg_90) / (math.sqrt(math.pi)))
c_120 = ((2 * avg_120) / (math.sqrt(math.pi)))
c_150 = ((2 * avg_150) / (math.sqrt(math.pi)))
c_180 = ((2 * avg_180) / (math.sqrt(math.pi)))
c_210 = ((2 * avg_210) / (math.sqrt(math.pi)))
c_240 = ((2 * avg_240) / (math.sqrt(math.pi)))
c_270 = ((2 * avg_270) / (math.sqrt(math.pi)))
c_300 = ((2 * avg_300) / (math.sqrt(math.pi)))
c_330 = ((2 * avg_330) / (math.sqrt(math.pi)))
c_360 = ((2 * avg_360) / (math.sqrt(math.pi)))

# Arranging wind speeds
wind = []
for i in range(0, 1000):
    wind.append(i * 0.05)

# Creating a list to include pdf and cdf results
weibull_pdf_result = []
weibull_cdf_result = []

# Creating a for-loop to include all results
for i in range(0, 1000):
    weibull_pdf_result.append(Weibull.weibull_pdf(wind[i], weibull_k, weibull_c))
    weibull_cdf_result.append(Weibull.weibull_cdf(wind[i], weibull_k, weibull_c))

# Creating dataframe
data = {"wind speed": wind, "weibull_pdf": weibull_pdf_result, "weibull_cdf": weibull_cdf_result}
dataframe = pd.DataFrame(data=data)

# Creating the power curve for our turbine
Pr = 15000000  # Constant power from 10.6 m/s to 25 m/s
cut_in = 3  # Cut-in wind speed limit
rated = 10.59  # rated wind speed limit
n = 2  # Exponent in the power formula
w = sp.Symbol('w')
w_s = np.linspace(cut_in, rated, 1000)

# Create a list of wind speeds with an interval of 0.05 m/s
wind_speeds = np.arange(0, 50, 0.05)

# Calculate power values using the provided formula
power_values = []

for wind_speed in wind_speeds:
    if cut_in <= wind_speed <= rated:
        power = Pr * ((wind_speed ** n - cut_in ** n) / (rated ** n - cut_in ** n))
    elif rated < wind_speed <= 25:
        power = Pr
    else:
        power = 0

    power_values.append(power)

# Create a table (list of lists) with wind speeds and corresponding power values
wind_power_df = pd.DataFrame({'wind_speed_150m': wind_speeds, 'power': power_values})

# Print the table
for row in wind_power_df:
    print(row)

# Wind speed vs power graph
# plt.figure(figsize=(10, 5))
# plt.plot(wind_speeds, power_values, label='Power')
# plt.title('Wind Speed vs Power')
# plt.xlabel('Wind Speed (m/s)')
# plt.ylabel('Power (10^7 Watts)')
# plt.legend()
# plt.grid(True)
# plt.show()

# Defining parameters for our wind turbine
u = [0, 3, 10.59, 25, 30]  # cut-in, rated, cut-out
ct = [0, 8 / 9, 8 / 9, 1 / 9, 0]  # Thrust coefficient
power = [0, 0, 15000, 15000, 0]  # Power in kW corresponding to u and ct

# The IEA 15MW Wind Turbine
my_wt = WindTurbine(name='MyWT',
                    diameter=240,
                    hub_height=150,
                    powerCtFunction=PowerCtTabular(u, power, 'kW', ct))

# Parameters defined previously
f = [dir_30, dir_60, dir_90, dir_120, dir_150, dir_180, dir_210, dir_240, dir_270, dir_300, dir_330,
     dir_360]  # Avg direction for each direction
A = [c_30, c_60, c_90, c_120, c_150, c_180, c_210, c_240, c_270, c_300, c_330, c_360]  # Weibull scale parameter, c
k = [k_30, k_60, k_90, k_120, k_150, k_180, k_210, k_240, k_270, k_300, k_330, k_360]  # Weibull shape parameter, k
wd = np.linspace(0, 360, len(f), endpoint=False)  # Wind direction
ti = hourly_standard_deviation / hourly_wind_speed_150m_mean  # Turbulence intensity calculated

# Site with wind direction dependent weibull distributed wind speed
site = XRSite(
    ds=xr.Dataset(data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ti},
                  coords={'wd': wd}))

# Comparison between the IEA15MW turbine and a pre-defined 10MW turbine by DTU
windTurbines = my_wt
dtu10mw = DTU10MW()

# Define grid-based layout parameters
# turbine_spacing_x = 1000  # Spacing between turbines in meters on the x-axis
# turbine_spacing_y = 1000  # Spacing between turbines in meters on the y-axis
# num_rows = 4  # Number of rows
# num_cols = 5  # Number of columns
offset_x = 0  # Starting point on x-axis
offset_y = 0  # Starting point on y-axis

# Compute turbine coordinates
initial_x = 0  # Initial x-coordinate for the first turbine
initial_y = 0  # Initial y-coordinate for the first turbine
x_coord_tab = []
y_coord_tab = []
initial_wind_direction = []
#
#
# # Generate turbine coordinates
# turbine_coordinates = []
# for row in range(num_rows):
#     if offset_x == 0:
#         a = 0# offset_x = turbine_spacing_x / 2
#     else:
#         offset_x = 0
#     for col in range(num_cols):
#         x_coord = offset_x + col * turbine_spacing_x
#         y_coord = initial_y + row * turbine_spacing_y
#         x_coord_tab.append(x_coord)
#         y_coord_tab.append(y_coord)
#         initial_wind_direction.append(45)  # The direction can be changed
#         turbine_coordinates.append((x_coord, y_coord))
#
# # Print the generated turbine coordinates
# for idx, coord in enumerate(turbine_coordinates):
#     print(f"Turbine {idx + 1}: ({coord[0]}, {coord[1]})")

if menu == 1:
    offset_x = initial_x
elif menu == 2:
    offset_x = turbine_spacing_x / 2

# Generate turbine coordinates
turbine_coordinates = []
for row in range(num_rows):
    # Calculate row offset based on the menu choice and row number
    if (menu == 1 and row % 2 == 0) or (menu == 2 and row % 2 != 0):  # First and third row starts at (0, 0)
        row_offset = initial_x
    else:
        row_offset = offset_x  # Second and fourth row starts at turbine_spacing_x/2
    for col in range(num_cols):
        x_coord = row_offset + col * turbine_spacing_x
        y_coord = initial_y + row * turbine_spacing_y
        x_coord_tab.append(x_coord)
        y_coord_tab.append(y_coord)
        initial_wind_direction.append(338)  # The direction can be changed
        turbine_coordinates.append((x_coord, y_coord))
    if menu == 2:  # Swap row_offset for the next row in zigzag layout
        row_offset = initial_x if row % 2 != 0 else offset_x

# Print the generated turbine coordinates
for idx, coord in enumerate(turbine_coordinates):
    print(f"Turbine {idx + 1}: ({coord[0]}, {coord[1]})")


# Different Wake Deficit Models
noj = NOJ(site, windTurbines)
BG = BastankhahGaussian(site, windTurbines)

# Creating a simulation result variable
simulationResult = noj(x_coord_tab, y_coord_tab)
simulationresult_BG = BG(x_coord_tab, y_coord_tab)

# Initialize the highest value to a low value
highest_value_noj = 0

# Iterate over each turbine
for x in range(20):
    # Calculate the AEP of the turbine
    turbine_aep = simulationResult.Power.sel(method='nearest', wt=x, wd=338).sum().values / 1e6
    print(f'Turbine {x + 1} Power: ', turbine_aep)
    # Update highest value
    if highest_value_noj < turbine_aep:
        highest_value_noj = turbine_aep

# Initialize the highest value to a low value
highest_value_bg = 0

# Iterate over each turbine
for x in range(20):
    # Calculate the AEP of the turbine
    turbine_aep_bg = simulationresult_BG.Power.sel(method='nearest', wt=x, wd=338).sum().values / 1e6
    print(f'Turbine {x + 1} Power: ', turbine_aep_bg)
    # Update highest value
    if highest_value_bg < turbine_aep_bg:
        highest_value_bg = turbine_aep_bg

# Calculating total AEP with and without wake loss for NOJ
aep_with_wake_loss_NOJ = np.round(simulationResult.aep().sum().data, 0)  # AEP with wake loss
aep_without_wake_loss_NOJ = np.round(simulationResult.aep(with_wake_loss=False).sum().data, 0)  # AEP without wake loss
total_wake_loss_GWh_NOJ = np.round((aep_without_wake_loss_NOJ - aep_with_wake_loss_NOJ),
                                0)  # Total wake loss in GWh
total_wake_loss_per_NOJ = np.round(((1 - (aep_with_wake_loss_NOJ / aep_without_wake_loss_NOJ)) * 100),
                                0)  # Total wake loss in %
print('Total AEP Without Wake Loss NOJ:', aep_without_wake_loss_NOJ, 'GWh')  # Printing the result
print('Total AEP With Wake Loss NOJ:', aep_with_wake_loss_NOJ, 'GWh')  # Printing the result
print('Total wake loss NOJ:', total_wake_loss_GWh_NOJ, 'GWh')  # Printing the result
print('Total Wake Loss NOJ:', total_wake_loss_per_NOJ, '%')  # Printing the result
# print("Total AEP of NOJ: %f GWh" % simulationResult.aep().sum())

# Calculating total AEP with and without wake loss for Bastankhah Gaussian Model
aep_with_wake_loss_BG = np.round(simulationresult_BG.aep().sum().data, 0)  # AEP with wake loss
aep_without_wake_loss_BG = np.round(simulationresult_BG.aep(with_wake_loss=False).sum().data, 0)  # AEP without wake loss
total_wake_loss_GWh_BG = np.round((aep_without_wake_loss_BG - aep_with_wake_loss_BG),
                               0)  # Total wake loss in GWh
total_wake_loss_per_BG = np.round(((1 - (aep_with_wake_loss_BG / aep_without_wake_loss_BG)) * 100),
                               0)  # Total wake loss in %
print('Total AEP Without Wake Loss BG:', aep_without_wake_loss_BG, 'GWh')  # Printing the result
print('Total AEP With Wake Loss BG:', aep_with_wake_loss_BG, 'GWh')  # Printing the result
print('total wake loss BG:', total_wake_loss_GWh_BG, 'GWh')  # Printing the result
print('Total Wake Loss BG:', total_wake_loss_per_BG, '%')  # Printing the result
# print("Total AEP of GB: %f GWh" % simulationresult_BG.aep().sum())

# Categorizing the different wind turbines
wts = WindTurbines.from_WindTurbine_lst([dtu10mw, my_wt])
types = wts.types()
print("Name:\t\t%s" % "\t".join(wts.name(types)))
print('Diameter[m]\t%s' % "\t".join(map(str, wts.diameter(type=types))))
print('Hubheigt[m]\t%s' % "\t".join(map(str, wts.hub_height(type=types))))

# Plot 1, comparison of power curve for both turbines
ws = np.arange(3, 25)
plt.xlabel('Wind speed [m/s]')
plt.ylabel('Power [kW]')

for t in types:
    plt.plot(ws, wts.power(ws, type=t) * 1e-3, '.-', label=wts.name(t))
plt.legend(loc=1)
plt.show()

# Plot 2, comparison of thrust coefficient curve for both turbines
plt.xlabel('Wind speed [m/s]')
plt.ylabel('CT [-]')

for t in types:
    plt.plot(ws, wts.ct(ws, type=t), '.-', label=wts.name(t))
plt.legend(loc=1)

# NOJ Plots

# Plot 3, scatterplot showing AEP of each turbine using the NOJ model
plt.figure()
aep = simulationResult.aep()
windTurbines.plot(x_coord_tab, y_coord_tab)
c = plt.scatter(x_coord_tab, y_coord_tab, c=aep.sum(['wd', 'ws']))
plt.colorbar(c, label='AEP [GWh]')
plt.title('NOJ: AEP of each turbine')
plt.xlabel('x [m]')
plt.ylabel('[m]')

# Plot 3, scatterplot showing AEP of each turbine using the Bastankhah Gaussian model
plt.figure()
aep_bg = simulationresult_BG.aep()
windTurbines.plot(x_coord_tab, y_coord_tab)
c = plt.scatter(x_coord_tab, y_coord_tab, c=aep_bg.sum(['wd', 'ws']))
plt.colorbar(c, label='AEP [GWh]')
plt.title('BG: AEP of each turbine')
plt.xlabel('x [m]')
plt.ylabel('[m]')

# Plot 4, Total AEP vs Wind speed plot using the NOJ model
plt.figure()
aep.sum(['wt', 'wd']).plot()
plt.xlabel("Wind speed [m/s]")
plt.ylabel("AEP [GWh]")
plt.title('NOJ: AEP vs wind speed')

# Plot 4, Total AEP vs Wind speed plot using the Bastankhah Gaussian model
plt.figure()
aep_bg.sum(['wt', 'wd']).plot()
plt.xlabel("Wind speed [m/s]")
plt.ylabel("AEP [GWh]")
plt.title('BG: AEP vs wind speed')

# Plot 5, Total AEP vs Wind direction plot using the NOJ model
plt.figure()
aep.sum(['wt', 'ws']).plot()
plt.xlabel("Wind direction [deg]")
plt.ylabel("AEP [GWh]")
plt.title('NOJ: AEP vs wind direction')

# Plot 5, Total AEP vs Wind direction plot using the Bastankhah Gaussian model
plt.figure()
aep_bg.sum(['wt', 'ws']).plot()
plt.xlabel("Wind direction [deg]")
plt.ylabel("AEP [GWh]")
plt.title('BG: AEP vs wind direction')

# Plot 6, Wind farm simulation with wake loss using the NOJ model
wind_speed = 10
wind_direction = 338

plt.figure()
plt.figure(figsize=(5, 4))
flow_map = simulationResult.flow_map(ws=wind_speed, wd=wind_direction)
flow_map.plot_wake_map()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('NOJ: Wake map for' + f' {wind_speed} m/s and {wind_direction} deg')
plt.tight_layout()

# Plot 6, Wind farm simulation with wake loss using the Bastankhah Gaussian model
plt.figure()
plt.figure(figsize=(5, 4))
flow_map = simulationresult_BG.flow_map(ws=wind_speed, wd=wind_direction)
flow_map.plot_wake_map()
plt.xlabel('x [m]')
plt.ylabel('y [m]')
plt.title('BG: Wake map for' + f' {wind_speed} m/s and {wind_direction} deg')

plt.show()

# Wind Rose parameters
N = 500
ws1 = hourly_wind_speed_150m
wd1 = hourly_wind_direction_100m
ax = WindroseAxes.from_ax()
ax.bar(wd1, ws1, normed=True, opening=0.8, edgecolor="white")
ax.set_legend()


# Weibull pdf plot
def pdf_general(wind, weibull_k, weibull_c):
    return (weibull_k / weibull_c) * ((wind / weibull_c) ** (weibull_k - 1)) * math.exp(
        -((wind / weibull_c) ** weibull_k))


# Creating a definition for the CDF calculation
def cdf_general(wind, weibull_k, weibull_c):
    return 1 - math.exp(-((wind / weibull_c) ** weibull_k))


# Calculation the pdf and CDF results
pdf_result = [pdf_general(w, weibull_k, weibull_c) for w in wind]
cdf_result = [cdf_general(w, weibull_k, weibull_c) for w in wind]

# Plot PDF
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(wind, pdf_result, 'r-', label='PDF')
plt.xlabel('Wind Speed')
plt.ylabel('Probability Density')
plt.title('Weibull PDF')
plt.legend()

# Plot CDF
plt.subplot(1, 2, 2)
plt.plot(wind, cdf_result, 'b-', label='CDF')
plt.xlabel('Wind Speed')
plt.ylabel('Cumulative Probability')
plt.title('Weibull CDF')
plt.legend()

plt.tight_layout()

plt.show()

# January

# Hourly dataframe for january
January = [hourly_dataframe[0:744], hourly_dataframe[8760:9504], hourly_dataframe[17520:18264],
           hourly_dataframe[26304:27048], hourly_dataframe[35064:35808], hourly_dataframe[43824:44568],
           hourly_dataframe[52584:53328], hourly_dataframe[61368:62112], hourly_dataframe[70128:70872],
           hourly_dataframe[78888:79632], hourly_dataframe[87648:87672]]
# print("January:")
all_january_data = pd.concat(January)
january_mean = statistics.mean(all_january_data['wind_speed_150m'])  # Average wind speed
january_deviation = all_january_data['wind_speed_150m'] - january_mean  # Deviation
january_standard_deviation = (math.sqrt((sum(january_deviation ** 2)) / len(all_january_data)))  # Standard Deviation
january_weibull_k = ((january_standard_deviation / january_mean) ** (-1.090))  # Weibull shape parameter, k
january_weibull_c = ((2 * january_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c


# Creating a definition for the pdf calculation
def pdf_january(wind, january_weibull_k, january_weibull_c):
    return (january_weibull_k / january_weibull_c) * ((wind / january_weibull_c) ** (january_weibull_k - 1)) * math.exp(
        -((wind / january_weibull_c) ** january_weibull_k))


# Creating a definition for the CDF calculation
def cdf_january(wind, january_weibull_k, january_weibull_c):
    return 1 - math.exp(-((wind / january_weibull_c) ** january_weibull_k))


# Calculation the pdf and CDF results
january_pdf_result = [pdf_january(w, january_weibull_k, january_weibull_c) for w in wind]
january_cdf_result = [cdf_january(w, january_weibull_k, january_weibull_c) for w in wind]

# Calculating the CDF for both 3m/s and 25m/s
jan_cdf_3 = (january_cdf_result[61])
jan_cdf_25 = (january_cdf_result[501])

# Calculating the percentage of time in production
jan_time_in_production = jan_cdf_25 - jan_cdf_3

# Calculating the CDF at rated power
jan_rated_10 = (january_cdf_result[213])
jan_rated_25 = (january_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_jan = (jan_rated_25 - jan_rated_10) * 24

# Power produced
pdf_power_january = np.array(january_pdf_result) * np.array(power_values) * 720
cdf_power_january = np.array(january_cdf_result) * np.array(power_values) * 720

wind_power_january = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_january,
     'Power CDF': cdf_power_january})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_january, wind_speeds)

# Compute the total production over 30 days
production_30_jan = (integral_result * 720 + Pr * (jan_cdf_25 - jan_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_jan)

# February

February = [hourly_dataframe[744:1416], hourly_dataframe[9504:10176], hourly_dataframe[18264:18960],
            hourly_dataframe[27048:27720], hourly_dataframe[35808:36480], hourly_dataframe[44568:45240],
            hourly_dataframe[53328:54024], hourly_dataframe[62112:62784], hourly_dataframe[70872:71544],
            hourly_dataframe[79632:80304]]
# print("February:")
all_february_data = pd.concat(February)
february_mean = statistics.mean(all_february_data['wind_speed_150m'])  # Average wind speed
february_deviation = all_february_data['wind_speed_150m'] - february_mean  # Deviation
february_standard_deviation = (math.sqrt((sum(february_deviation ** 2)) / len(all_february_data)))  # Standard Deviation
february_weibull_k = ((february_standard_deviation / february_mean) ** (-1.090))  # Weibull shape parameter, k
february_weibull_c = ((2 * february_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_february(wind, february_weibull_k, february_weibull_c):
    return (february_weibull_k / february_weibull_c) * (
            (wind / february_weibull_c) ** (february_weibull_k - 1)) * math.exp(
        -((wind / february_weibull_c) ** february_weibull_k))

# Creating a definition for the CDF calculation
def cdf_february(wind, february_weibull_k, february_weibull_c):
    return 1 - math.exp(-((wind / february_weibull_c) ** february_weibull_k))

# Calculating the pdf and CDF results
february_pdf_result = [pdf_february(w, february_weibull_k, february_weibull_c) for w in wind]
february_cdf_result = [cdf_february(w, february_weibull_k, february_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
feb_cdf_3 = (february_cdf_result[61])
feb_cdf_25 = (february_cdf_result[501])

# Calculating time in production
feb_time_in_production = feb_cdf_25 - feb_cdf_3

# Calculating CDF at rated power
feb_rated_10 = (february_cdf_result[213])
feb_rated_25 = (february_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_feb = (feb_rated_25 - feb_rated_10) * 24

# Power produced
pdf_power_february = np.array(february_pdf_result) * np.array(power_values) * 720
cdf_power_february = np.array(february_cdf_result) * np.array(power_values) * 720

wind_power_february = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_february,
     'Power CDF': cdf_power_february})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_february, wind_speeds)

# Compute the total production over 30 days
production_30_feb = (integral_result * 720 + Pr * (feb_cdf_25 - feb_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_feb)

# March

march = [hourly_dataframe[1416:2160], hourly_dataframe[10176:10920], hourly_dataframe[18960:19704],
         hourly_dataframe[27720:28464], hourly_dataframe[36480:37224], hourly_dataframe[45240:45984],
         hourly_dataframe[45024:54768], hourly_dataframe[62784:63528], hourly_dataframe[71544:72288],
         hourly_dataframe[80304:81048]]
# print("March:")
all_march_data = pd.concat(march)
march_mean = statistics.mean(all_march_data['wind_speed_150m'])  # Average wind speed
march_deviation = all_march_data['wind_speed_150m'] - march_mean  # Deviation
march_standard_deviation = (math.sqrt((sum(march_deviation ** 2)) / len(all_march_data)))  # Standard deviation
march_weibull_k = ((march_standard_deviation / march_mean) ** (-1.090))  # Weibull shape parameter, k
march_weibull_c = ((2 * march_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_march(wind, march_weibull_k, march_weibull_c):
    return (march_weibull_k / march_weibull_c) * ((wind / march_weibull_c) ** (march_weibull_k - 1)) * math.exp(
        -((wind / march_weibull_c) ** march_weibull_k))

# Creating a definition for the CDF calculation
def cdf_march(wind, march_weibull_k, march_weibull_c):
    return 1 - math.exp(-((wind / march_weibull_c) ** march_weibull_k))

# Calculating the pdf and CDF results
march_pdf_result = [pdf_march(w, march_weibull_k, march_weibull_c) for w in wind]
march_cdf_result = [cdf_march(w, march_weibull_k, march_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
mar_cdf_3 = (march_cdf_result[61])
mar_cdf_25 = (march_cdf_result[501])

# Calculating time in production
mar_time_in_production = mar_cdf_25 - mar_cdf_3

# Calculating CDF at rated power
march_rated_10 = (march_cdf_result[213])
march_rated_25 = (march_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_march = (march_rated_25 - march_rated_10) * 24

# Power produced
pdf_power_march = np.array(march_pdf_result) * np.array(power_values) * 720
cdf_power_march = np.array(march_cdf_result) * np.array(power_values) * 720

wind_power_march = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_march, 'Power CDF': cdf_power_march})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_march, wind_speeds)

# Compute the total production over 30 days
production_30_mar = (integral_result * 720 + Pr * (mar_cdf_25 - march_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_mar)

# April

april = [hourly_dataframe[2160:2880], hourly_dataframe[10920:11640], hourly_dataframe[19704:20424],
         hourly_dataframe[28464:29184], hourly_dataframe[37224:37944], hourly_dataframe[45984:46704],
         hourly_dataframe[54768:55488], hourly_dataframe[63528:64248], hourly_dataframe[72288:73008],
         hourly_dataframe[81048:81768]]
# print("April:")
all_april_data = pd.concat(april)
april_mean = statistics.mean(all_april_data['wind_speed_150m'])  # Average wind speed
april_deviation = all_april_data['wind_speed_150m'] - april_mean  # Deviation
april_standard_deviation = (math.sqrt((sum(april_deviation ** 2)) / len(all_april_data)))  # Standard deviation
april_weibull_k = ((april_standard_deviation / april_mean) ** (-1.090))  # Weibull shape parameter, k
april_weibull_c = ((2 * april_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_april(wind, april_weibull_k, april_weibull_c):
    return (april_weibull_k / april_weibull_c) * ((wind / april_weibull_c) ** (april_weibull_k - 1)) * math.exp(
        -((wind / april_weibull_c) ** april_weibull_k))

# Creating a definition for the CDF calculation
def cdf_april(wind, april_weibull_k, april_weibull_c):
    return 1 - math.exp(-((wind / april_weibull_c) ** april_weibull_k))

# Calculating the pdf and CDF results
april_pdf_result = [pdf_april(w, april_weibull_k, april_weibull_c) for w in wind]
april_cdf_result = [cdf_april(w, april_weibull_k, april_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
apr_cdf_3 = (april_cdf_result[61])
apr_cdf_25 = (april_cdf_result[501])

# Calculating time in production
apr_time_in_production = apr_cdf_25 - apr_cdf_3

# Calculating CDF at rated power
april_rated_10 = (april_cdf_result[213])
april_rated_25 = (april_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_april = (april_rated_25 - april_rated_10) * 24

# Power produced
pdf_power_april = np.array(april_pdf_result) * np.array(power_values) * 720
cdf_power_april = np.array(april_cdf_result) * np.array(power_values) * 720

wind_power_april = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_april, 'Power CDF': cdf_power_april})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_april, wind_speeds)

# Compute the total production over 30 days
production_30_apr = (integral_result * 720 + Pr * (apr_cdf_25 - april_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_apr)

# May

may = [hourly_dataframe[2880:3624], hourly_dataframe[11640:12384], hourly_dataframe[20424:21168],
       hourly_dataframe[29184:29928], hourly_dataframe[37944:38688], hourly_dataframe[46704:47448],
       hourly_dataframe[55488:56232], hourly_dataframe[64248:64992], hourly_dataframe[73008:73752],
       hourly_dataframe[81768:82512]]
# print("May:")
all_may_data = pd.concat(may)
may_mean = statistics.mean(all_may_data['wind_speed_150m'])  # Average wind speed
may_deviation = all_may_data['wind_speed_150m'] - may_mean  # Deviation
may_standard_deviation = (math.sqrt((sum(may_deviation ** 2)) / len(all_may_data)))  # Standard Deviation
may_weibull_k = ((may_standard_deviation / may_mean) ** (-1.090))  # Weibull shape parameter, k
may_weibull_c = ((2 * may_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_may(wind, may_weibull_k, may_weibull_c):
    return (may_weibull_k / may_weibull_c) * ((wind / may_weibull_c) ** (may_weibull_k - 1)) * math.exp(
        -((wind / may_weibull_c) ** may_weibull_k))

# Creating a definition for the CDF calculation
def cdf_may(wind, may_weibull_k, may_weibull_c):
    return 1 - math.exp(-((wind / may_weibull_c) ** may_weibull_k))

# Calculating the pdf and CDF results
may_pdf_result = [pdf_may(w, may_weibull_k, may_weibull_c) for w in wind]
may_cdf_result = [cdf_may(w, may_weibull_k, may_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
may_cdf_3 = (may_cdf_result[61])
may_cdf_25 = (may_cdf_result[501])

# Calculating time in production
may_time_in_production = may_cdf_25 - may_cdf_3

# Calculating CDF at rated power
may_rated_10 = (may_cdf_result[213])
may_rated_25 = (may_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_may = (may_rated_25 - may_rated_10) * 24

# Power produced
pdf_power_may = np.array(may_pdf_result) * np.array(power_values) * 720
cdf_power_may = np.array(may_cdf_result) * np.array(power_values) * 720

wind_power_may = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_may, 'Power CDF': cdf_power_may})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_may, wind_speeds)

# Compute the total production over 30 days
production_30_may = (integral_result * 720 + Pr * (may_cdf_25 - may_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_may)

# June

june = [hourly_dataframe[3624:4344], hourly_dataframe[12384:13104], hourly_dataframe[21168:21888],
        hourly_dataframe[29928:30648], hourly_dataframe[38688:39408], hourly_dataframe[47448:48168],
        hourly_dataframe[56232:56952], hourly_dataframe[64992:65712], hourly_dataframe[73752:74472],
        hourly_dataframe[82512:83232]]
# print("June:")
all_june_data = pd.concat(june)
june_mean = statistics.mean(all_june_data['wind_speed_150m'])  # Average wind speed
june_deviation = all_june_data['wind_speed_150m'] - june_mean  # Deviation
june_standard_deviation = (math.sqrt((sum(june_deviation ** 2)) / len(all_june_data)))  # Standard Deviation
june_weibull_k = ((june_standard_deviation / june_mean) ** (-1.090))  # Weibull shape parameter, k
june_weibull_c = ((2 * june_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_june(wind, june_weibull_k, june_weibull_c):
    return (june_weibull_k / june_weibull_c) * ((wind / june_weibull_c) ** (june_weibull_k - 1)) * math.exp(
        -((wind / june_weibull_c) ** june_weibull_k))

# Creating a definition for the CDF calculation
def cdf_june(wind, june_weibull_k, june_weibull_c):
    return 1 - math.exp(-((wind / june_weibull_c) ** june_weibull_k))

# Calculating the pdf and CDF results
june_pdf_result = [pdf_june(w, june_weibull_k, june_weibull_c) for w in wind]
june_cdf_result = [cdf_june(w, june_weibull_k, june_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
jun_cdf_3 = (june_cdf_result[61])
jun_cdf_25 = (june_cdf_result[501])

# Calculating time in production
jun_time_in_production = jun_cdf_25 - jun_cdf_3

# Calculating CDF at rated power
june_rated_10 = (june_cdf_result[213])
june_rated_25 = (june_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_june = (june_rated_25 - june_rated_10) * 24

# Power produced
pdf_power_june = np.array(june_pdf_result) * np.array(power_values) * 720
cdf_power_june = np.array(june_cdf_result) * np.array(power_values) * 720

wind_power_june = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_june, 'Power CDF': cdf_power_june})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_june, wind_speeds)

# Compute the total production over 30 days
production_30_jun = (integral_result * 720 + Pr * (jun_cdf_25 - june_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_jun)

# July

july = [hourly_dataframe[4344:5088], hourly_dataframe[13104:13848], hourly_dataframe[21888:22362],
        hourly_dataframe[30648:31392], hourly_dataframe[39408:40152], hourly_dataframe[48168:48912],
        hourly_dataframe[56952:57696], hourly_dataframe[65712:66456], hourly_dataframe[74472:75216],
        hourly_dataframe[83232:83976]]
# print("July:")
all_july_data = pd.concat(july)
july_mean = statistics.mean(all_july_data['wind_speed_150m'])  # Average wind speed
july_deviation = all_july_data['wind_speed_150m'] - july_mean  # Deviation
july_standard_deviation = (math.sqrt((sum(july_deviation ** 2)) / len(all_july_data)))  # Standard Deviation
july_weibull_k = ((july_standard_deviation / july_mean) ** (-1.090))  # Weibull shape parameter, k
july_weibull_c = ((2 * july_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_july(wind, july_weibull_k, july_weibull_c):
    return (july_weibull_k / july_weibull_c) * ((wind / july_weibull_c) ** (july_weibull_k - 1)) * math.exp(
        -((wind / july_weibull_c) ** july_weibull_k))

# Creating a definition for the CDF calculation
def cdf_july(wind, july_weibull_k, july_weibull_c):
    return 1 - math.exp(-((wind / july_weibull_c) ** july_weibull_k))

# Calculating the pdf and CDF results
july_pdf_result = [pdf_july(w, july_weibull_k, july_weibull_c) for w in wind]
july_cdf_result = [cdf_july(w, july_weibull_k, july_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
jul_cdf_3 = (july_cdf_result[61])
jul_cdf_25 = (july_cdf_result[501])

# Calculating time in production
jul_time_in_production = jul_cdf_25 - jul_cdf_3

# Calculating CDF at rated power
july_rated_10 = (july_cdf_result[213])
july_rated_25 = (july_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_july = (july_rated_25 - july_rated_10) * 24

# Power produced
pdf_power_july = np.array(july_pdf_result) * np.array(power_values) * 720
cdf_power_july = np.array(july_cdf_result) * np.array(power_values) * 720

wind_power_july = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_july, 'Power CDF': cdf_power_july})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_july, wind_speeds)

# Compute the total production over 30 days
production_30_jul = (integral_result * 720 + Pr * (jul_cdf_25 - july_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_jul)

# August

august = [hourly_dataframe[5088:5832], hourly_dataframe[13848:14592], hourly_dataframe[22362:23376],
          hourly_dataframe[31392:32136], hourly_dataframe[40152:40896], hourly_dataframe[48912:49656],
          hourly_dataframe[57696:58440], hourly_dataframe[66456:67200], hourly_dataframe[75216:75960],
          hourly_dataframe[83976:84720]]
# print("August:")
all_august_data = pd.concat(august)
august_mean = statistics.mean(all_august_data['wind_speed_150m'])  # Average wind speed
august_deviation = all_august_data['wind_speed_150m'] - august_mean  # Deviation
august_standard_deviation = (math.sqrt((sum(august_deviation ** 2)) / len(all_august_data)))  # Standard Deviation
august_weibull_k = ((august_standard_deviation / august_mean) ** (-1.090))  # Weibull shape parameter, k
august_weibull_c = ((2 * august_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_august(wind, august_weibull_k, august_weibull_c):
    return (august_weibull_k / august_weibull_c) * ((wind / august_weibull_c) ** (august_weibull_k - 1)) * math.exp(
        -((wind / august_weibull_c) ** august_weibull_k))

# Creating a definition for the CDF calculation
def cdf_august(wind, august_weibull_k, august_weibull_c):
    return 1 - math.exp(-((wind / august_weibull_c) ** august_weibull_k))

# Calculating the pdf and CDF results
august_pdf_result = [pdf_august(w, august_weibull_k, august_weibull_c) for w in wind]
august_cdf_result = [cdf_august(w, august_weibull_k, august_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
aug_cdf_3 = (august_cdf_result[61])
aug_cdf_25 = (august_cdf_result[501])

# Calculating time in production
aug_time_in_production = aug_cdf_25 - aug_cdf_3

# Calculating CDF at rated power
aug_rated_10 = (august_cdf_result[213])
aug_rated_25 = (august_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_august = (aug_rated_25 - aug_rated_10) * 24

# Power produced
pdf_power_august = np.array(august_pdf_result) * np.array(power_values) * 720
cdf_power_august = np.array(august_cdf_result) * np.array(power_values) * 720

wind_power_august = pd.DataFrame({'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_august,
                                  'Power CDF': cdf_power_august})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_august, wind_speeds)

# Compute the total production over 30 days
production_30_aug = (integral_result * 720 + Pr * (aug_cdf_25 - aug_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_aug)

# September

september = [hourly_dataframe[5832:6552], hourly_dataframe[14592:15312], hourly_dataframe[23376:24096],
             hourly_dataframe[32136:32856], hourly_dataframe[40896:41616], hourly_dataframe[49656:50376],
             hourly_dataframe[58440:59160], hourly_dataframe[67200:67920], hourly_dataframe[75960:76680],
             hourly_dataframe[84720:85440]]
# print("September:")
all_september_data = pd.concat(september)
september_mean = statistics.mean(all_september_data['wind_speed_150m'])  # Average wind speed
september_deviation = all_september_data['wind_speed_150m'] - september_mean  # Deviation
september_standard_deviation = (math.sqrt((sum(september_deviation ** 2)) / len(all_september_data)))  # Standard Deviation
september_weibull_k = ((september_standard_deviation / september_mean) ** (-1.090))  # Weibull shape parameter, k
september_weibull_c = ((2 * september_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_september(wind, september_weibull_k, september_weibull_c):
    return (september_weibull_k / september_weibull_c) * (
            (wind / september_weibull_c) ** (september_weibull_k - 1)) * math.exp(
        -((wind / september_weibull_c) ** september_weibull_k))

# Creating a definition for the CDF calculation
def cdf_september(wind, september_weibull_k, september_weibull_c):
    return 1 - math.exp(-((wind / september_weibull_c) ** september_weibull_k))

# Calculating the pdf and CDF results
september_pdf_result = [pdf_september(w, september_weibull_k, september_weibull_c) for w in wind]
september_cdf_result = [cdf_september(w, september_weibull_k, september_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
sep_cdf_3 = (september_cdf_result[61])
sep_cdf_25 = (september_cdf_result[501])

# Calculating time in production
sep_time_in_production = sep_cdf_25 - sep_cdf_3

# Calculating CDF at rated power
sep_rated_10 = (september_cdf_result[213])
sep_rated_25 = (september_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_sep = (sep_rated_25 - sep_rated_10) * 24

# Power produced
pdf_power_september = np.array(september_pdf_result) * np.array(power_values) * 720
cdf_power_september = np.array(september_cdf_result) * np.array(power_values) * 720

wind_power_september = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_september,
     'Power CDF': cdf_power_september})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_september, wind_speeds)

# Compute the total production over 30 days
production_30_sep = (integral_result * 720 + Pr * (sep_cdf_25 - sep_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_sep)

# October

october = [hourly_dataframe[6552:7296], hourly_dataframe[15312:16056], hourly_dataframe[24096:24840],
           hourly_dataframe[32856:33600], hourly_dataframe[41616:42360], hourly_dataframe[50376:51120],
           hourly_dataframe[59160:59904], hourly_dataframe[67920:68664], hourly_dataframe[76680:77424],
           hourly_dataframe[85440:86184]]
# print("October:")
all_october_data = pd.concat(october)
october_mean = statistics.mean(all_october_data['wind_speed_150m'])  # Average wind speed
october_deviation = all_october_data['wind_speed_150m'] - october_mean  # Deviation
october_standard_deviation = (math.sqrt((sum(october_deviation ** 2)) / len(all_october_data)))  # Standard Deviation
october_weibull_k = ((october_standard_deviation / october_mean) ** (-1.090))  # Weibull shape parameter, k
october_weibull_c = ((2 * october_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_october(wind, october_weibull_k, october_weibull_c):
    return (october_weibull_k / october_weibull_c) * ((wind / october_weibull_c) ** (october_weibull_k - 1)) * math.exp(
        -((wind / october_weibull_c) ** october_weibull_k))

# Creating a definition for the CDF calculation
def cdf_october(wind, october_weibull_k, october_weibull_c):
    return 1 - math.exp(-((wind / october_weibull_c) ** october_weibull_k))

# Calculating the pdf and CDF results
october_pdf_result = [pdf_october(w, october_weibull_k, october_weibull_c) for w in wind]
october_cdf_result = [cdf_october(w, october_weibull_k, october_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
oct_cdf_3 = (october_cdf_result[61])
oct_cdf_25 = (october_cdf_result[501])

# Calculating time in production
oct_time_in_production = oct_cdf_25 - oct_cdf_3

# Calculating CDF at rated power
oct_rated_10 = (october_cdf_result[213])
oct_rated_25 = (october_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_oct = (oct_rated_25 - oct_rated_10) * 24

# Power produced
pdf_power_october = np.array(october_pdf_result) * np.array(power_values) * 720
cdf_power_october = np.array(october_cdf_result) * np.array(power_values) * 720

wind_power_october = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_october,
     'Power CDF': cdf_power_october})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_october, wind_speeds)

# Compute the total production over 30 days
production_30_oct = (integral_result * 720 + Pr * (oct_cdf_25 - oct_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_oct)

# November

november = [hourly_dataframe[7296:8016], hourly_dataframe[16056:16776], hourly_dataframe[24840:25560],
            hourly_dataframe[33600:34320], hourly_dataframe[42360:43080], hourly_dataframe[51120:51840],
            hourly_dataframe[59904:60624], hourly_dataframe[68664:69384], hourly_dataframe[77424:78144],
            hourly_dataframe[86184:86904]]
# print("November:")
all_november_data = pd.concat(november)
november_mean = statistics.mean(all_november_data['wind_speed_150m'])  # Average wind speed
november_deviation = all_november_data['wind_speed_150m'] - november_mean  # Deviation
november_standard_deviation = (math.sqrt((sum(november_deviation ** 2)) / len(all_november_data)))  # Standard Deviation
november_weibull_k = ((november_standard_deviation / november_mean) ** (-1.090))  # Weibull shape parameter, k
november_weibull_c = ((2 * november_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_november(wind, november_weibull_k, november_weibull_c):
    return (november_weibull_k / november_weibull_c) * (
            (wind / november_weibull_c) ** (november_weibull_k - 1)) * math.exp(
        -((wind / november_weibull_c) ** november_weibull_k))

# Creating e definition for the CDF calculation
def cdf_november(wind, november_weibull_k, november_weibull_c):
    return 1 - math.exp(-((wind / november_weibull_c) ** november_weibull_k))

# Calculating the pdf and CDF results
november_pdf_result = [pdf_november(w, november_weibull_k, november_weibull_c) for w in wind]
november_cdf_result = [cdf_november(w, november_weibull_k, november_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
nov_cdf_3 = (november_cdf_result[61])
nov_cdf_25 = (november_cdf_result[501])

# Calculating time in production
nov_time_in_production = nov_cdf_25 - nov_cdf_3

# Calculating CDF at rated power
nov_rated_10 = (november_cdf_result[213])
nov_rated_25 = (november_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_nov = (nov_rated_25 - nov_rated_10) * 24

# Power produced
pdf_power_november = np.array(november_pdf_result) * np.array(power_values) * 720
cdf_power_november = np.array(november_cdf_result) * np.array(power_values) * 720

wind_power_november = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_november,
     'Power CDF': cdf_power_november})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_november, wind_speeds)

# Compute the total production over 30 days
production_30_nov = (integral_result * 720 + Pr * (nov_cdf_25 - nov_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_nov)

# December

december = [hourly_dataframe[8016:8760], hourly_dataframe[16776:17520], hourly_dataframe[25560:26304],
            hourly_dataframe[34320:35064], hourly_dataframe[43080:43824], hourly_dataframe[51840:52584],
            hourly_dataframe[60624:61368], hourly_dataframe[69384:70128], hourly_dataframe[78144:78888],
            hourly_dataframe[86904:87648]]
# print("December:")
all_december_data = pd.concat(december)
december_mean = statistics.mean(all_december_data['wind_speed_150m'])  # Average wind speed
december_deviation = all_december_data['wind_speed_150m'] - december_mean  # Deviation
december_standard_deviation = (math.sqrt((sum(december_deviation ** 2)) / len(all_december_data)))  # Standard Deviation
december_weibull_k = ((december_standard_deviation / december_mean) ** (-1.090))  # Weibull shape parameter, k
december_weibull_c = ((2 * december_mean) / (math.sqrt(math.pi)))  # Weibull scale parameter, c

# Creating a definition for the pdf calculation
def pdf_december(wind, december_weibull_k, december_weibull_c):
    return (december_weibull_k / december_weibull_c) * (
            (wind / december_weibull_c) ** (december_weibull_k - 1)) * math.exp(
        -((wind / december_weibull_c) ** december_weibull_k))

# Creating a definition for the CDF calculation
def cdf_december(wind, december_weibull_k, december_weibull_c):
    return 1 - math.exp(-((wind / december_weibull_c) ** december_weibull_k))

# Calculating the pdf and CDF results
december_pdf_result = [pdf_december(w, december_weibull_k, december_weibull_c) for w in wind]
december_cdf_result = [cdf_december(w, december_weibull_k, december_weibull_c) for w in wind]

# Calculating the CDF at both 3m/s and 25m/s
dec_cdf_3 = (december_cdf_result[61])
dec_cdf_25 = (december_cdf_result[501])

# Calculating time in production
dec_time_in_production = dec_cdf_25 - dec_cdf_3

# Calculating CDF at rated power
dec_rated_10 = (december_cdf_result[213])
dec_rated_25 = (december_cdf_result[501])

# Number of hours operating at rated power
daily_rated_power_dec = (dec_rated_25 - dec_rated_10) * 24

# Power produced
pdf_power_december = np.array(december_pdf_result) * np.array(power_values) * 720
cdf_power_december = np.array(december_cdf_result) * np.array(power_values) * 720

wind_power_december = pd.DataFrame(
    {'wind_speed_150m': wind_speeds, 'power': power_values, 'Power pdf': pdf_power_december,
     'Power CDF': cdf_power_december})

# Compute the integral using trapezoidal rule
integral_result = trapezoid(pdf_power_december, wind_speeds)

# Compute the total production over 30 days
production_30_dec = (integral_result * 720 + Pr * (dec_cdf_25 - dec_rated_10) * 720) / (10 ** 6)
# print("Estimated production over 30 days:", production_30_nov)

# Results power

total_production = (
        production_30_jan + production_30_feb + production_30_mar + production_30_apr + production_30_may + production_30_jun + production_30_jul + production_30_aug + production_30_sep + production_30_oct + production_30_nov + production_30_dec)
# print("Total production: ", total_production)

info = ['Percentage of time in production', 'Number of hours operating at rated power per day [h]',
        'Estimated production over 30 days [GWh]']
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
jan = [jan_time_in_production, daily_rated_power_jan, production_30_jan]
feb = [feb_time_in_production, daily_rated_power_feb, production_30_feb]
mar = [mar_time_in_production, daily_rated_power_march, production_30_mar]
apr = [apr_time_in_production, daily_rated_power_april, production_30_apr]
mai = [may_time_in_production, daily_rated_power_may, production_30_may]
jun = [jun_time_in_production, daily_rated_power_june, production_30_jun]
jul = [jul_time_in_production, daily_rated_power_july, production_30_jul]
aug = [aug_time_in_production, daily_rated_power_august, production_30_aug]
sep = [sep_time_in_production, daily_rated_power_sep, production_30_sep]
oct = [oct_time_in_production, daily_rated_power_oct, production_30_oct]
nov = [nov_time_in_production, daily_rated_power_nov, production_30_nov]
dec = [dec_time_in_production, daily_rated_power_dec, production_30_dec]

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot bar diagrams for each category
for i in range(3):
    axs[i].bar(months, [jan[i], feb[i], mar[i], apr[i], mai[i], jun[i], jul[i], aug[i], sep[i], oct[i], nov[i], dec[i]])
    axs[i].set_ylabel(info[i])

# Plot Monthly performance metrics
plt.suptitle('Monthly Performance Metrics')

# Show the plots
plt.show()

Results = {
    'January': jan,
    'February': feb,
    'March': mar,
    'April': apr,
    'May': mai,
    'June': jun,
    'July': jul,
    'August': aug,
    'September': sep,
    'October': oct,
    'November': nov,
    'December': dec,
}

results = pd.DataFrame(Results)
# print(results)