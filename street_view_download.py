import urllib

# url2 = 'https://maps.googleapis.com/maps/api/streetview?size=400x400&location=40.720032,-73.988354&fov=90&heading=235&pitch=50'
cont = 'y'
count = 100
while (cont == 'y') :
	lat = raw_input("latitude : ")
	longi = raw_input("longitude : ")
	fov = raw_input("Field of view : ")
	heading = raw_input("heading : ")
	pitch = raw_input("pitch : ")

	url1 = 'https://maps.googleapis.com/maps/api/streetview?size=400x400&location='+lat+','+longi+'&fov='+fov+'&heading='+heading+'&pitch='+pitch
	urllib.urlretrieve(url1,'/home/madhan/ImageMatching/third_task/Images/img'+str(count)+'.jpeg')
	
	count += 1
	cont = raw_input("Continue? (y/n)")
