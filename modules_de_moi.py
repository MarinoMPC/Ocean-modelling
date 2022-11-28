#Mes modules pour faire des trucs tavu
import os 
import numpy as np

import sys
sys.path.insert(0,'/home2/datahome/mmenard/lagrangian/')
import tools_for_croco as to

########### PARAMETERS ##############
time_step_1f=10
time_step_sim=12     # in hours
dfile=2

def files_listing(txt_files):
    l=[]
    with open(txt_files,'r') as f:
        for line in f:
            l.append(line[:-1])
    return(l)


def recover(file_txt):
    with open(file_txt+'.txt', "r") as file:
        for l in file:
            data=[]
            elt=''
            for i in l[1:]:
                if i!=',' and i!=']':
                    elt+=i
                else:
                    data.append(float(elt))
                    elt=''
    return(np.array(data))


def particle_counter(plon,continuous_injection=True):
    Nt=plon.shape[0]
    nb_part=np.zeros(Nt)
    for i in range(Nt):
        nb_part[i]=np.nansum(plon[i,:]!=0.0)
    part_flux=nb_part[0]
    nb_part_max=nb_part[-1]
    if continuous_injection==True:
        dt_injection=np.nanmin(np.where(nb_part>part_flux))
        period_injection=time_step_sim*dfile*dt_injection/24
        return(int(part_flux),period_injection,int(nb_part_max))
    else:
        return(part_flux)


def fusion(file1,file2):
    txt1,txt2=open(file1+'.txt', 'r'),open(file2+'.txt', 'r')
    s=0
    fus=''
    for caca1 in txt1:
        for pipi1 in caca1[:-1]:
            fus+=pipi1
    fus+=','
    for caca2 in txt2:
        for pipi2 in caca2[1:]:
            fus+=pipi2
    title=file1[:-9]+file2[-9:]
    with open(title + '.txt','w') as output :
        output.write(fus)


def annual_mean(data,years_number,nb_points_year=365):
    data_sep=np.zeros((years_number,nb_points_year))
    for i in range(years_number):
        data_sep[i,:]=data[i*nb_points_year:(i+1)*nb_points_year]
        data_bar=np.nanmean(data_sep,axis=0)
    return(data_bar)


def month_generator(nb_months,month_freq,starting_month):

    '''
    nb_months : total number of months
    month_freq : frequence for the ticks. Equal to 1 when all monthes appear
    starting_month : integer which illustrates the start of the month_list
    '''
    
    
    pos,ticks=[],[]
    for i in range(starting_month,nb_months+starting_month,month_freq):
        if i%12==1:
            ticks.append('Jan')
        elif i%12==2:
            ticks.append('Feb')
        elif i%12==3:
            ticks.append('Mar')
        elif i%12==4:
            ticks.append('Apr')
        elif i%12==5:
            ticks.append('May')
        elif i%12==6:
            ticks.append('Jun')
        elif i%12==7:
            ticks.append('Jul')
        elif i%12==8:
            ticks.append('Aug')
        elif i%12==9:
            ticks.append('Sep')
        elif i%12==10:
            ticks.append('Oct')
        elif i%12==11:
            ticks.append('Nov')
        elif i%12==0:
            ticks.append('Dec')
    return(ticks)


def is_in(lat_part,lon_part,box):
    ymin,ymax,xmin,xmax=box[0],box[1],box[2],box[3]
    return((lat_part<ymax) & (lat_part>ymin) & (lon_part<xmax) & (lon_part>xmin))


def truncate(filename, count, ignore_newlines=True):
    """
    Truncates last `count` characters of a text file encoded in UTF-8.
    :param filename: The path to the text file to read
    :param count: Number of UTF-8 characters to remove from the end of the file
    :param ignore_newlines: Set to true, if the newline character at the end of the file should be ignored
    """
    with open(filename, 'rb+') as f:
        last_char = None

        size = os.fstat(f.fileno()).st_size

        offset = 1
        chars = 0
        while offset <= size:
            f.seek(-offset, os.SEEK_END)
            b = ord(f.read(1))

            if ignore_newlines:
                if b == 0x0D or b == 0x0A:
                    offset += 1
                    continue

            if b & 0b10000000 == 0 or b & 0b11000000 == 0b11000000:
                # This is the first byte of a UTF8 character
                chars += 1
                if chars == count:
                    # When `count` number of characters have been found, move current position back
                    # with one byte (to include the byte just checked) and truncate the file
                    f.seek(-1, os.SEEK_CUR)
                    f.truncate()
                    return
            offset += 1


def sigma_to_z_4d(data,eta,grid,depths=-np.linspace(0,200,21)):
    
    Nt,Nz,Ny,Nx=data.shape[0],depths.size,data.shape[2],data.shape[3]
    data_z=np.zeros((Nt,Nz,Ny,Nx))
    
    hc = grid.hc
    Cs_r = grid.Cs_r
    Cs_w = grid.Cs_w
    topo=grid.variables['h'][latmin:latmax,lonmin:lonmax] # on the rho grid

    for t in range(Nt):
        z_r0 =  to.zlevs(topo,eta[t].values, hc, Cs_r, Cs_w)[0]
        data_z[t] = to.vinterps(data[t].values,z_r0,depths,topo=topo,cubic=1)
    return(data_z)



def sigma_to_z_3d(data,eta,grid,latmin,latmax,lonmin,lonmax,depths=-np.linspace(0,200,21)):
    
    hc = grid.hc
    Cs_r = grid.Cs_r
    Cs_w = grid.Cs_w
    topo=grid.variables['h'][latmin:latmax,lonmin:lonmax] # on the rho grid
    
    z_r0 =  to.zlevs(topo,eta.values, hc, Cs_r, Cs_w)[0]
    data_z = to.vinterps(data.values,z_r0,depths,topo=topo,cubic=1)
    return(data_z)

def sigma_to_z_merid_sec(data,eta,grid,latmin_sec,latmax_sec,lon_sec,depths=-np.linspace(0,200,21)):

    Nt,Nz,Ny=data.shape[0],depths.size,data.shape[2]
    data_z=np.zeros((Nt,Nz,Ny))

    hc = grid.hc
    Cs_r = grid.Cs_r
    Cs_w = grid.Cs_w
    topo=grid.variables['h'][latmin_sec:latmax_sec,lon_sec] # on the rho grid

    z_r0 =  to.zlevs_sec(topo.values,eta.values, hc, Cs_r, Cs_w)[0]
    data_z = to.vinterps(data.values,z_r0,depths,topo=topo,cubic=1)
    return(data_z)


def array_in_txt_1D(array1d,txt_title,path='',printing=True):
    '''
    array1d : 1D array to write in the txt file (numpy array)
    txt_title : name of the txt file in which array1d will be written, write it without '.txt' (str)
    path : path where it will be saved. Example : '/home2/datahome/zzidane/coupe_du_monde_98/' (str)
    '''
    l=array1d.tolist()
    with open(path+txt_title+'.txt','w') as file:
        file.write(str(l))
    if printing==True:
        print('Written in '+path+txt_title+'.txt')
        
   

def spherical_distance(lon1,lat1,lon2,lat2,Re=6.371e+6):
    '''
    lon1,lat1 : coordinates of the first point on the Earth, in degrees
    lon2,lat2 : coordinates of the second point on the Earth, in degrees
    Re (optional) : Earth radius
    '''
    inter=np.sin((lat1-lat2)/2*np.pi/180)**2+np.cos(lat1*np.pi/180)*np.cos(lat2*np.pi/180)*np.sin((lon1-lon2)/2*np.pi/180)**2
    return(2*Re*np.arcsin(inter)**0.5)


def truncate(pdepth):
    nb_part_max=np.nansum(pdepth[-1]!=0.0)
    return(pdepth[:,:nb_part_max])

def ticks_place(Ndays):
    nb_complete_years=int(Ndays//365)
    days_months=np.array([0,31,28,31,30,31,30,31,31,30,31,30,31])
    nb_days=[]
    for i in range(nb_complete_years):
        for j in days_months:
            nb_days.append(j)        
    
    nb_days_remaining=Ndays%365
    month=1
    while days_months[month]<nb_days_remaining:
        nb_days.append(days_months[month])
        nb_days_remaining=nb_days_remaining-days_months[month]
        month+=1
    
    ticks_place=np.cumsum(nb_days)
    return(ticks_place) 

def month_conv(number,year):
    if number==int(1):
        return('JAN',31)
    elif number==int(2) and year%4==0 and year!=2000:
        return('FEB',29)
    elif number==int(2):
        return('FEB',28)
    elif number==int(3):
        return('MAR',31)
    elif number==int(4):
        return('APR',30)
    elif number==int(5):
        return('MAY',31)
    elif number==int(6):
        return('JUN',30)
    elif number==int(7):
        return('JUL',31)
    elif number==int(8):
        return('AUG',31)
    elif number==int(9):
        return('SEP',30)
    elif number==int(10):
        return('OCT',31)
    elif number==int(11):
        return('NOV',30)
    elif number==int(12):
        return('DEC',31)   

def title(file_sim,output_freq):
    '''
    file is the name of the file 
    Nt : number of time steps in the file 
    output_freq : number of hours between two outputs, needs to be a divider of 24
    doesn't take in account leap years yet
    '''
    title=[]
    Nt_daily=24//output_freq
    year=np.int(file_sim[-24:-20])
    month=np.int(file_sim[-19:-17])
    day=file_sim[-16:-14]
    title=day+' '+month_conv(month,year)[0]+' '+str(year)
    return(title)


def uv_rho_surf(sim,latmin,latmax,lonmin,lonmax):

    u=sim['u'][:,-1,latmin-1:latmax,lonmin:lonmax]
    v=sim['v'][:,-1,latmin:latmax,lonmin-1:lonmax]
    
    u_rho=.5*(u.values[:,1:,:]+u.values[:,:-1,:])
    v_rho=.5*(v.values[:,:,1:]+v.values[:,:,:-1])

    return(np.squeeze(u_rho),np.squeeze(v_rho))


def uv_rho_depth(sim,grid,latmin,latmax,lonmin,lonmax,depth=-300):
    if depth==0:
        return(uv_rho_surf(sim,latmin,latmax,lonmin,lonmax))
    else:
        u=sim['u'][:,:,latmin-1:latmax,lonmin:lonmax]
        v=sim['v'][:,:,latmin:latmax,lonmin-1:lonmax]
        eta=sim['zeta'][:,latmin:latmax,lonmin:lonmax]

        u_rho_sigma=.5*(u.values[:,:,1:,:]+u.values[:,:,:-1,:])
        v_rho_sigma=.5*(v.values[:,:,:,1:]+v.values[:,:,:,:-1])

        hc = sim.hc
        Cs_r = grid.Cs_r
        Cs_w = grid.Cs_w
        topo=grid.variables['h'][latmin:latmax,lonmin:lonmax] # on the rho grid*
        
        u_rho=np.zeros((u_rho_sigma.shape[0],u_rho_sigma.shape[2],u_rho_sigma.shape[3]))
        v_rho=np.zeros((u_rho_sigma.shape[0],u_rho_sigma.shape[2],u_rho_sigma.shape[3]))

        for t in range(u.shape[0]):
            
            z_r =  to.zlevs(topo,eta[t].values, hc, Cs_r, Cs_w)[0]

            u_rho[t]=to.vinterp(u_rho_sigma[t],z_r,depth,topo=topo,cubic=1)
            v_rho[t]=to.vinterp(v_rho_sigma[t],z_r,depth,topo=topo,cubic=1)
    
        return(np.squeeze(u_rho),np.squeeze(v_rho))


def volume_rho(grid,depths,indices_zoom):
    Re=6.371e+6 
    latmin,latmax,lonmin,lonmax=indices_zoom[0],indices_zoom[1],indices_zoom[2],indices_zoom[3]
    lon_rho=grid['lon_rho'][latmin:latmax,lonmin:lonmax]
    lat_rho=grid['lat_rho'][latmin:latmax,lonmin:lonmax]
    lon_u=grid['lon_u'][latmin:latmax+1,lonmin:lonmax]
    lat_u=grid['lat_u'][latmin:latmax+1,lonmin:lonmax]
    lon_v=grid['lon_v'][latmin:latmax,lonmin:lonmax+1]
    lat_v=grid['lat_v'][latmin:latmax,lonmin:lonmax+1]
    Nz,Ny,Nx=depths.size,lat_rho.shape[0],lat_rho.shape[1]
    
    dx_rho,dy_rho,dz_rho=np.ones((Nz,Ny,Nx)),np.ones((Nz,Ny,Nx)),np.ones((Nz,Ny,Nx))
    
    new_depths=0.5*(depths[1:]+depths[:-1])
    new_depths2=np.zeros(depths.size+1)
    new_depths2[1:-1]=new_depths
    new_depths2[0]   =depths[0]
    new_depths2[-1]  =depths[-1]
    
    dx=(lon_v.values[:,1:]-lon_v.values[:,:-1])*np.pi/180*Re*np.cos(lat_v.values[:,:-1]*np.pi/180)
    dy=(lat_u.values[1:,:]-lat_u.values[:-1,:])*Re*np.pi/180
    dz=abs(new_depths2[1:]-new_depths2[:-1])
    
    
    dx_rho[:]=dx
    dy_rho[:]=dy
    for z in range(Nz):
        dz_rho[z,:,:]=dz[z]
        
    return(dx_rho*dy_rho*dz_rho)

def bool_box(box,lon,lat):
    latmin,latmax,lonmin,lonmax=box[0],box[1],box[2],box[3]
    return((lon<=lonmax) & (lon>=lonmin) & (lat<=latmax) & (lat>=latmin))
