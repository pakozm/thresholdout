-- This is an adaptation of Dwork et al's supplemenary code for the paper "The
-- reusable holdout: Preserving validity in adaptive data analysis". It was
-- originally hosted at
--
-- https://www.sciencemag.org/content/suppl/2015/08/05/349.6248.636.DC1/aaa9375_SupportingFile_Other_seq5_v1.py
--
-- Experiments for Thresholdhout
-- Fast implementation of Thresholdout specific to the experiment.
-- Thresholdout with threshold = 4/sqrt(n), tolerance = 1/sqrt(n)
-- Signal: 20 variables with 6/sqrt(n) bias toward the label
--
-- Instead of linear classifier we use here an ANN with one hidden
-- layer. Instead of variable selection, we have done here topology
-- selection. Variable selection is performed by means of L1 regularization.

local aprilann = require "aprilann"
local gp = require "april_tools.gnuplot"()

local rng  = random(24825)
local rng4 = random(59268)

local function sample(n,d)
  return stats.dist.normal():sample(rng, n*d):rewrap(n, d)
end

local function createdataset(n,d)
  local X = sample(n,d+1)
  X[{':',d+1}]:sign()
  return X
end

local function createnosignaldata(n,d)
  -- Data points sampled from Gaussian distribution. Class labels random and
  -- uniform.
  local X_train = createdataset(n,d)
  local X_holdout = createdataset(n,d)
  local X_test = createdataset(n,d)
  return X_train,X_holdout,X_test
end

local function addbias(n,d,X,nbiased)
  local bias = 6.0/math.sqrt(n)
  local b = matrix(1,nbiased):zeros()
  for i=1,n do
    b[{}] = X:get(i,d+1) * bias
    X[{i,{1,nbiased}}]:axpy(1.0, b)
  end
  return X
end

local function createhighsignaldata(n,d)
  -- Data points are random Gaussian vectors. Class labels are random and
  -- uniform. First nbiased are biased with bias towards the class label

  local X_train,X_holdout,X_test = createnosignaldata(n,d)
  
  -- Add correlation with the sign
  local nbiased   = 20
  local X_train   = addbias(n, d, X_train,   nbiased)
  local X_holdout = addbias(n, d, X_holdout, nbiased)
  local X_test    = addbias(n, d, X_test,    nbiased)
  
  return X_train, X_holdout, X_test  
end

local function createpairds(X,center,scale)
  local X = X:clone()
  X:select(2, X:dim(2)):clamp(0,1)
  local Y = X[{':',{1,X:dim(2)-1}}]
  Y[{}],center,scale = stats.standardize(Y,{center=center,scale=scale})
  local ds     = dataset.matrix(X)
  local in_ds  = dataset.split(ds, 1, ds:patternSize()-1)
  local out_ds = dataset.split(ds, ds:patternSize(), ds:patternSize())
  return in_ds,out_ds,center,scale
end

local function createtraintable(X, rng)
  local in_ds,out_ds,center,scale = createpairds(X)
  return {
    input_dataset  = in_ds,
    output_dataset = out_ds,
    shuffle        = rng,
  },center,scale
end

local function createvaltable(X,center,scale)
  local in_ds,out_ds = createpairds(X,center,scale)
  return {
    input_dataset  = in_ds,
    output_dataset = out_ds,
    loss = ann.loss.zero_one(0, math.log(0.5)),
  }
end

local function train(n,d,hrange,X_train,X_holdout,X_test,evaluate_func)
  local vals = {}
  for _,h in ipairs(hrange) do
    local rng2 = random(8952)
    local rng3 = random(9285)
    local model = ann.mlp.all_all.generate("%d inputs %d logistic 1 log_logistic"%{d,h})
    local trainer = trainable.supervised_trainer(model, ann.loss.cross_entropy(),
                                                 32, ann.optimizer.adadelta())
    trainer:build()
    trainer:randomize_weights{ random=rng2, inf=-0.1, sup=0.1 }
    -- trainer:set_option("learning_rate", 0.1)
    -- trainer:set_option("L1_norm", 0.0001)
    -- trainer:set_option("weight_decay", 0.0)
    local criterion = trainable.stopping_criteria.make_max_epochs_wo_imp_absolute(10)
    local pocket = trainable.train_holdout_validation{
      max_epochs = 10,
      min_epochs = 100,
      stopping_criterion = criterion,
    }
    local train_table,center,scale = createtraintable(X_train, rng3)
    local train_table_accuracy = createvaltable(X_train,center,scale)
    local val_table_accuracy = createvaltable(X_holdout,center,scale)
    local test_table_accuracy = createvaltable(X_test,center,scale)
    while pocket:execute(function()
        trainer:train_dataset(train_table)
        local tr_loss = trainer:validate_dataset(train_table_accuracy)
        local va_loss = trainer:validate_dataset(val_table_accuracy)
        return trainer,tr_loss,va_loss
    end) do
      print(pocket:get_state_string(), trainer:norm2("w.*"), trainer:norm2("b.*"))
    end
    local best = pocket:get_state_table().best
    local tr,va,te = evaluate_func(best,train_table_accuracy,
                                   val_table_accuracy,test_table_accuracy)
    table.insert(vals, tr)
    table.insert(vals, va)
    table.insert(vals, te)
  end
  return vals
end

local function runClassifier(n,d,hrange,X_train,X_holdout,X_test)
  -- Variable selection and basic ANN classifier on synthetic data.
  local tolerance = 1.0/math.sqrt(n)
  local threshold = 4.0/math.sqrt(n)
  local vals = train(n,d,hrange,X_train,X_holdout,X_test,
                     function(best,tr_tbl,va_tbl,te_tbl)
                       local tr = 1 - best:validate_dataset(tr_tbl)
                       local va = 1 - best:validate_dataset(va_tbl)
                       local te = 1 - best:validate_dataset(te_tbl)
                       return tr,va,te
  end)
  -- Compute values using Thresholdout
  local noisy_vals = train(n,d,hrange,X_train,X_holdout,X_test,
                           function(best,tr_tbl,va_tbl,te_tbl)
                             local tr = 1 - best:validate_dataset(tr_tbl)
                             local va = 1 - best:validate_dataset(va_tbl)
                             local te = 1 - best:validate_dataset(te_tbl)
                             if math.abs(tr-va) < threshold + rng4:randNorm(0,tolerance) then
                               va = tr
                             else
                               va = va + rng4:randNorm(0,tolerance)
                             end
                             return tr,va,te
  end)
  return vals, noisy_vals
end

local function repeatexp(n,d,hrange,reps,datafn)
  -- Repeat experiment specified by fn for reps steps
  local vallist = {}
  local vallist2 = {}
  for r=1,reps do
    print("Repetition:", r)
    local X_train,X_holdout,X_test = datafn(n,d)
    local vals,vals2 = runClassifier(n,d,hrange,X_train,X_holdout,X_test)
    table.insert(vallist, matrix(1,#vals,3,vals))
    table.insert(vallist2, matrix(1,#vals2,3,vals2))
  end
  return vallist, vallist2
end

local function runandplot(n,d,hrange,reps,datafn,plotname)
  local vallist,vallist2 = repeatexp(n,d,hrange,reps,datafn)
  local std,mean = stats.std(matrix.join(1,vallist), 1):squeeze()
  local std2,mean2 = stats.std(matrix.join(1,vallist2), 1):squeeze()
  local hrange = matrix(#hrange,1,hrange)
  local Y = matrix.join(2, hrange, mean, std, mean2, std2)
  Y:toTabFilename(plotname)
end

local reps = 100
local n, d = 10000, 10000
local hrange = { 2,4,8,16,32,64,128,256,512 }
-- local hrange = { 8 }

runandplot(n,d,hrange,reps,createnosignaldata,"plot_no_signal")
runandplot(n,d,hrange,reps,createhighsignaldata,"plot_signal")
