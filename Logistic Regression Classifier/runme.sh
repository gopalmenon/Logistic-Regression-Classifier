#!/bin/bash
clear
javac -d bin -sourcepath src src/*.java
java -classpath "bin" LogisticRegressionClassifierClient
