import math
from django.shortcuts import render
from django.http import JsonResponse
import ee
from autogluon.tabular import TabularPredictor
import numpy as np
import pandas as pd
import os
import matplotlib.cm as cm
from datetime import datetime as dt
from branca.element import MacroElement
from jinja2 import Template
import folium
from folium import raster_layers
import json
import branca.colormap as cmb
import pickle
import warnings
warnings.filterwarnings('ignore')

#initialize earthengine
ee.Initialize()

# This class is redundant, use javascript space to do popup functions
class Popup(MacroElement):
    """
    When one clicks on a Map that contains a LatLngPopup,
    a popup is shown that displays the latitude and longitude of the pointer.

    """
    
    def __init__(self,array,bound):
        super().__init__()
        self._name = "Popup"
        json_data = json.dumps(array)
        boundary = json.dumps(bound)
        self._template = Template(
            """
                {% macro script(this, kwargs) %}
                    var {{this.get_name()}} = L.popup();
                    function latLngPop(e) {
                        var lat = e.latlng.lat.toFixed(4)
                        var lon = e.latlng.lng.toFixed(4)
                        {{this.get_name()}}
                            .setLatLng(e.latlng)
                            .setContent("Latitude: " + e.latlng.lat.toFixed(4) +
                                        "<br>Longitude: " + e.latlng.lng.toFixed(4)) 
                            .openOn({{this._parent.get_name()}});
                        var arr = JSON.parse('{{json_data}}');
                        var bound = JSON.parse('{{boundary}}')
                        if (lat>bound[1][0]){
                                if (lat<bound[0][0]){
                                    if(lon>bound[0][1]){
                                        if(lon<bound[1][1]){
                                            var pixel_val = getPixel(lat,lon)
                                            //console.log(pixel_val)
                                            {{this.get_name()}}.setLatLng(e.latlng).setContent("Latitude: " + e.latlng.lat.toFixed(4) +
                                        "<br>Longitude: " + e.latlng.lng.toFixed(4)+ "<br> Pixel Value: " + pixel_val.toFixed(4) ) 
                            .openOn({{this._parent.get_name()}});
                                        }
                                    }
                                }
                            }
                        function getPixel(lat,lon){
                            var image_width = bound[1][1] - bound[0][1] 
                            var image_height = bound[0][0] - bound[1][0]
                            var lat_ratio = (lat - bound[1][0]) / image_height
                            var lon_ratio = (lon - bound[0][1]) / image_width
                            var pixel_x = Math.floor(lat_ratio * arr.length)
                            var pixel_y = Math.floor(lon_ratio * arr[0].length)
                            return arr[(arr.length-pixel_x-1)][pixel_y]
                        }
                        }
                    {{this._parent.get_name()}}.on('click', latLngPop);
                {% endmacro %}
                """.replace('{{json_data}}', json_data).replace('{{boundary}}',boundary)
        )
        


root = os.getcwd()
infpath = os.path.join(root,"inflow.xlsx")
datapath = os.path.join(root,"data.xlsx")


inflow = pd.read_excel(infpath)
nitrate = pd.read_excel(datapath,sheet_name='Nitrate')
phosphate = pd.read_excel(datapath,sheet_name='Phosphate')
#print(inflow,nitrate,phosphate)
def param(img):
    bands = {
        'GREEN': img.select('B3'),
        'RED': img.select('B4'),
        'NIR': img.select('B5'),
        'SWIR1': img.select('B6'),
        "CA" : img.select('B1'),
        "SWIR2": img.select('B7'),
    }
    #chl = img.expression('1000*(0.004 - (3.362 * GREEN)+(6.065*RED)+(1.135*NIR)-(0.876*SWIR1))',bands ).rename('chl_a_mg/L')
    chl = img.expression('1000*((1.704 * CA)-(0.975*GREEN)+(0.287*NIR)-(2.095*SWIR2))',bands ).rename('chl_a_mg/L')
    tds = img.expression('10**(2.799 - (2.836 * GREEN)+(3.664*RED)+(0.265*NIR)+(2.407*SWIR1))', bands).rename('tds_mg/L')
    econ = img.expression('0.899 - (6.495 * GREEN)+(13.081*RED)+(1.053*NIR)+(1.439*SWIR1)', bands).rename('ec_mS/cm')
    return chl,tds,econ

def cloud(img):
    image = img.select('QA_PIXEL')
    cloudbitmask = (1<<3)
    cloudshadowbitmask = (1<<4)
    mask = image.bitwiseAnd(cloudbitmask).eq(0)
    mask2 = image.bitwiseAnd(cloudshadowbitmask).eq(0)
    watermask = image.bitwiseAnd((1<<7)).neq(0)
    img = img.updateMask(mask).updateMask(mask2).updateMask(watermask) 
    return img
def import_pred (model_name, label, input_data,shape):
    # Load the model using AutoGluon
    
    predictor = TabularPredictor.load(model_name)
    
    # Make predictions using the loaded model
    input_data = input_data[['Month', 'Chlorophyll a', 'Inflow', label]]  # Load or create your test data
    
    predictions = np.array(predictor.predict(input_data)).reshape(shape[0],shape[1])
    return predictions

def new_model(model_name,data,shape):
    model_path = os.path.join("FinalModels",model_name+".pkl")
    model = pickle.load(open(model_path,"rb"))
    return model.predict(data).reshape(shape[0],shape[1])

def locate(date,inflow,inval):
    date = date.split('-')
    if int(date[1])>10 and int(date[2])>10:
        date= '-'.join(date)
    else:
        if int(date[2])<10:
            date[2] = '0' + str(date[2])
        if int(date[1])<10:
            date[1] = '0'+str(date[1])
        date = '-'.join(date)
    #print(date)
    if(inval>-1):
        infl = inval
    else:    
        infl = float(inflow.loc[inflow['Dates'].astype(str).str.contains(date, case=False)]['mean'])
    #nitrate_val = float(nitrate.loc[nitrate['Dates'].astype(str).str.contains(date,case=False)]['SEG 1'])
    #phosphate_val = float(phosphate.loc[phosphate['Dates'].astype(str).str.contains(date,case=False)]['SEG 1'])
    #return [infl,nitrate_val,phosphate_val]
    return infl
colormap = cm.get_cmap('jet')
def map_color(value):
        if value == 0:
            return (0,0,0,0)
        else:
            rgba = colormap(value)
            rgb = [int(rgba[i] * 255) for i in range(3)]
            #print(rgb)
            return (rgb[0], rgb[1], rgb[2],1)
# Create your views here.

def view1(request):
    if request.method == "POST":
        boundary = [[request.POST['swLng'],request.POST['neLat']],[request.POST['swLng'],request.POST['swLat']],
                   [request.POST['neLng'],request.POST['swLat']],[request.POST['neLng'],request.POST['neLat']],
                   [request.POST['swLng'],request.POST['neLat']]]
        polygon = [[float(x) for x in row] for row in boundary]
        #print(request.POST['inf'],type(request.POST['inf']))
        infval = float(request.POST['inf'])
        aoi = ee.Geometry.Polygon(polygon,None,False)
        point = ee.Geometry.Point(78.147361,12.464021)
        year = request.POST['year']
        seasonValue = request.POST['season']
        #print(seasonValue)
        if seasonValue == 'winter':
            start = year + "-01-01"
            end = year + "-03-01"
        elif seasonValue == 'summer':
            start = year + "-03-01"
            end = year + "-06-01"
        elif seasonValue == 'swm':
            start = year + "-06-01"
            end = year + "-09-01"
        else:
            start = year + "-09-01"
            end = str(int(year)+1) + "-01-01"
        collection = (ee.ImageCollection("LANDSAT/LC08/C02/T1_TOA").filterDate(start, end).filterBounds(point)
        .filter(ee.Filter.lt('CLOUD_COVER', 20)).sort('CLOUD_COVER').select(['B1','B3','B4','B5','B6','B7','QA_PIXEL']))
        if collection.size().getInfo()<1:
            info = "No"
            return JsonResponse({"info":info})

        else:
            image_ee = collection.first()
            image_ee = image_ee.addBands(ee.Image([param(image_ee)]))
            date = ee.Date(image_ee.get('system:time_start')).format('Y-M-d').getInfo()
            arraysdict = (image_ee.select(['QA_PIXEL','chl_a_mg/L','tds_mg/L','ec_mS/cm']).sampleRectangle(region=aoi)).getInfo()['properties']
            chl_array = np.array(arraysdict["chl_a_mg/L"])
            tds_array = np.array(arraysdict["tds_mg/L"])
            ec_array = np.array(arraysdict["ec_mS/cm"])
            QA_pixel = np.array(arraysdict["QA_PIXEL"])
            #print(chl_array.shape,tds_array.shape)
            input_data = pd.DataFrame(chl_array.reshape(-1),columns=['Chlorophyll a'])
            months_columns = {'Apr':0, 'Aug':0, 'Dec':0, 'Feb':0, 'Jan':0, 'Jul':0, 'Jun':0, 'Mar':0, 'May':0, 'Nov':0, 'Oct':0, 'Sep':0}
            month_sel = (dt.strptime(date, "%Y-%m-%d")).strftime("%b")
            for key in months_columns.keys():
                if(key == month_sel):
                    months_columns[key] = 1
            #values = locate(date,nitrate,phosphate,inflow,infval)   commented out since its old model values
            #input_data['Month'] = (dt.strptime(date, "%Y-%m-%d")).strftime("%b");input_data['Nitrate'] = values[1];input_data['Phosphate'] = values[2];input_data['Inflow'] = values[0]
            #nci_array,pci_array = import_pred('nitrate', 'Nitrate' , input_data,chl_array.shape),import_pred('phosphate', 'Phosphate' , input_data,chl_array.shape)
            input_data['Inflow'] = locate(date,inflow,infval)
            input_data = input_data.assign(**months_columns)
            
            nci_array,pci_array = new_model("nitrate_model",input_data,chl_array.shape),new_model("phosphate_model",input_data,chl_array.shape)
            #print(nci_array)
            
            def mask(arr):
                water_mask = (arr & (1 << 7)) != 0
                cloud_mask = (arr & (1<<3)) == 0
                shadow = (arr & (1<<4)) == 0
                masked_array = np.where(water_mask,np.where(shadow,np.where(cloud_mask, arr, 0),0),0)
                return masked_array
            mask_arr = mask(QA_pixel)>0
            tds_arr = np.where(mask_arr,tds_array,0)
            ec_arr = np.where(mask_arr,ec_array,0)
            chl_arr = np.where(mask_arr,chl_array,0)
            nci_array = np.where(mask_arr,nci_array,0)
            pci_array = np.where(mask_arr,pci_array,0)
            

            # for outlier removal
            def w(arr):
                flattened_arr = arr.flatten()
                flattened_arr = flattened_arr[flattened_arr != 0]
                # Calculate the lower and upper bounds for outliers
                lower_bound = np.percentile(flattened_arr, 1.5)
                upper_bound = np.percentile(flattened_arr, 99.9)
                #print(lower_bound,upper_bound)
                # Remove outliers
                result = np.where((arr >= lower_bound) & (arr <= upper_bound), arr, 0)

                return result
            
            def r(arr):
                flattened_arr = arr.flatten()
                median = np.mean(flattened_arr)
                q1 = np.percentile(flattened_arr, 25)
                q3 = np.percentile(flattened_arr, 75)
                iqr = q3 - q1
                # Calculate the threshold based on median + 1.5 * IQR
                threshold = median + 1.5 * iqr
                # Replace right outliers with zeros
                result = np.where(arr > threshold, 0, arr)
                return result
            # nci_array = w(nci_array)
            # pci_array = w(pci_array)
            # tds_arr = w(tds_arr)
            # ec_arr = w(ec_arr)
            # chl_arr = w(chl_arr)
            chl_arr_mean = np.mean(chl_arr[chl_arr!=0])
            TSI = 9.81 * (math.log(chl_arr_mean)) +30.6
            #print(chl_arr_mean,TSI)
            def filterminmax(arr):
                filtered_numbers = [num for num in arr if num != 0]
                return min(filtered_numbers),max(filtered_numbers)
            
            min_tds,max_tds = filterminmax(tds_arr.flatten())
            min_ec,max_ec = filterminmax(ec_arr.flatten())
            min_chl,max_chl = filterminmax(chl_arr.flatten())
            min_nci,max_nci = filterminmax(nci_array.flatten())
            min_pci,max_pci = filterminmax(pci_array.flatten())

            mean_tds = np.mean(tds_arr[tds_arr!=0])
            mean_ec = np.mean(ec_arr[ec_arr!=0])
            mean_nci = np.mean(nci_array[nci_array!=0])
            mean_pci = np.mean(pci_array[pci_array!=0])

            #print(min_chl,max_chl)
            #print(min_nci,max_nci,min_pci,max_pci)
            # ------Masking------
            
            bounds = [(max(polygon)[1], min(polygon)[0]), (min(polygon)[1], max(polygon)[0])]
            color_tds = lambda value: map_color(value) if value ==0 else map_color((value - min_tds) / (max_tds - min_tds))
            color_ec = lambda value: map_color(value) if value ==0 else map_color((value - min_ec) / (max_ec - min_ec))
            color_chl = lambda value: map_color(value) if value ==0 else map_color((value - min_chl) / (max_chl - min_chl))
            color_nci = lambda value: map_color(value) if value ==0 else map_color((value - min_nci) / (max_nci - min_nci))
            color_pci = lambda value: map_color(value) if value ==0 else map_color((value - min_pci) / (max_pci - min_pci))
            image_overlay_3 = raster_layers.ImageOverlay(tds_arr, bounds=bounds,colormap=color_tds)
            image_overlay_4 = raster_layers.ImageOverlay(ec_arr, bounds=bounds,colormap=color_ec)
            image_overlay_0 = raster_layers.ImageOverlay(chl_arr, bounds=bounds,colormap=color_chl)
            image_overlay_1 = raster_layers.ImageOverlay(nci_array, bounds=bounds,colormap=color_nci)
            image_overlay_2 = raster_layers.ImageOverlay(pci_array,bounds=bounds,colormap=color_pci)
            #print(bounds)
            #Map part
            center = [(max(polygon)[1] + min(polygon)[1]) / 2, (min(polygon)[0] + max(polygon)[0]) / 2]  # Example center coordinate
            zoom_level = 13.5 # Example zoom level
            

            j = folium.Map(location=center,zoom_start=zoom_level)
            k = folium.Map(location=center,zoom_start=zoom_level)
            l = folium.Map(location=center,zoom_start=zoom_level)
            m = folium.Map(location=center, zoom_start=zoom_level)
            n = folium.Map(location=center, zoom_start=zoom_level)
            
            image_overlay_3.add_to(j)
            Popup(tds_arr.tolist(),bounds).add_to(j)
            color_scale_tds = cmb.LinearColormap(colors=[cm.jet(x) for x in np.linspace(0, 1, 256)], vmin=min_tds, vmax=max_tds)
            j.add_child(color_scale_tds)
            map_html_3 = j._repr_html_()

            image_overlay_4.add_to(k)
            Popup(ec_arr.tolist(),bounds).add_to(k)
            color_scale_ec = cmb.LinearColormap(colors=[cm.jet(x) for x in np.linspace(0, 1, 256)], vmin=min_ec, vmax=max_ec)
            k.add_child(color_scale_ec)
            map_html_4 = k._repr_html_()

            image_overlay_0.add_to(l)
            Popup(chl_arr.tolist(),bounds).add_to(l)
            color_scale_chl = cmb.LinearColormap(colors=[cm.jet(x) for x in np.linspace(0, 1, 256)], vmin=min_chl, vmax=max_chl)
            l.add_child(color_scale_chl)
            map_html_0 = l._repr_html_()

            image_overlay_1.add_to(m)
            Popup(nci_array.tolist(),bounds).add_to(m)
            color_scale_nci = cmb.LinearColormap(colors=[cm.jet(x) for x in np.linspace(0, 1, 256)], vmin=min_nci, vmax=max_nci)
            m.add_child(color_scale_nci)
            map_html_1 = m._repr_html_()
            
            image_overlay_2.add_to(n)
            Popup(pci_array.tolist(),bounds).add_to(n)
            color_scale_pci = cmb.LinearColormap(colors=[cm.jet(x) for x in np.linspace(0, 1, 256)], vmin=min_pci, vmax=max_pci)
            n.add_child(color_scale_pci)
            map_html_2 = n._repr_html_()
            if TSI<40:
                typeval= "Oligotrophic"
            elif TSI>39 and TSI<50:
                typeval = "Mesotrophic"
            elif TSI>49 and TSI<60:
                typeval = "Eutrophic"
            else:
                typeval = "Hypereutrophic"
            TSI = str(round(TSI,3)) +" - "+ typeval
            info = "yes"
            chl_mean,tds_mean,ec_mean,nci_mean,pci_mean = str(round(chl_arr_mean,3)),str(round(mean_tds,3)), str(round(mean_ec,3)),str(round(mean_nci,3)),str(round(mean_pci,3))
            jsonres = {
                "chlarr":info,
                'map0':map_html_0,
                'map1':map_html_1,
                'map2':map_html_2,
                'map3':map_html_3,
                'map4':map_html_4,
                'TSI':TSI,
                'mean_chl':chl_mean,
                'mean_tds':tds_mean,
                'mean_ec':ec_mean,
                'mean_nci':nci_mean,
                'mean_pci':pci_mean
                }
            return JsonResponse(jsonres)
    else:
        info = "null"
        return render(request,"index.html",context={})


