# OTDM Lab 2: Dual formulation of Support Vector Machine
# Marcel, Mengxue
# Autumn 2021


# Parameters
param n; 				# rows
param m; 				# columns
param nu;              	# tradeoff

param y {1..m};        	# response value
param A {1..m,1..n};   	# feature values

# Variables
var lambda {1..m} >= 0, <= nu;


# Dual formulation
maximize dual:
	sum{i in {1..m}}lambda[i] 
	-(1/2)*sum{i in {1..m}, j in {1..m}}lambda[i]*y[i]*lambda[j]*y[j]*(sum{k in {1..n}}A[i,k]*A[j,k]);
	
subject to c1:
	sum{i in {1..m}}(lambda[i]*y[i]) = 0;