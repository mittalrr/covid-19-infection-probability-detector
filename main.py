from flask import Flask,  render_template, request
app = Flask(__name__)


import pickle

filename = 'model.pkl'
file = open(filename,'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
	if request.method == "POST":
		# print(request.form)
		mydict = request.form
		fever = int(mydict['fever'])
		age = int(mydict['age'])
		pain = int(mydict['pain'])
		runnynose = int(mydict['runnynose'])
		diffbreath = int(mydict['diffbreath'])

		inputfeatures = [fever,pain,age,runnynose,diffbreath]
		infprob = clf.predict_proba([inputfeatures])[0][1]
		print("\n",infprob,"\n")

		return render_template('results.html', inf=round(infprob*100))
	return render_template('index.html')
	# return 'Hello, World!'+str(infprob)


if __name__ == '__main__':
	app.run(debug=True)