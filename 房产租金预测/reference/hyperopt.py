
from __future__ import print_function
import lightgbm as lgb
import sklearn
import numpy
import hyperopt
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
import colorama
import numpy as np

N_HYPEROPT_PROBES = 500
HYPEROPT_ALGO = tpe.suggest  #  tpe.suggest OR hyperopt.rand.suggest

# ----------------------------------------------------------

colorama.init()

# ---------------------------------------------------------------------

def get_lgb_params(space):
    lgb_params = dict()
    lgb_params['boosting_type'] = space['boosting_type'] if 'boosting_type' in space else 'gbdt'
    lgb_params['objective'] = 'regression'
    lgb_params['metric'] = 'rmse'
    lgb_params['learning_rate'] = space['learning_rate']
    lgb_params['num_leaves'] = int(space['num_leaves'])
    lgb_params['min_data_in_leaf'] = int(space['min_data_in_leaf'])
    lgb_params['min_sum_hessian_in_leaf'] = space['min_sum_hessian_in_leaf']
    lgb_params['max_depth'] = -1
    lgb_params['lambda_l1'] = space['lambda_l1'] if 'lambda_l1' in space else 0.0
    lgb_params['lambda_l2'] = space['lambda_l2'] if 'lambda_l2' in space else 0.0
    lgb_params['max_bin'] = int(space['max_bin']) if 'max_bin' in space else 256
    lgb_params['feature_fraction'] = space['feature_fraction']
    lgb_params['bagging_fraction'] = space['bagging_fraction']
    lgb_params['bagging_freq'] = int(space['bagging_freq']) if 'bagging_freq' in space else 1
    lgb_params['nthread'] = 4
    return lgb_params

# ---------------------------------------------------------------------

obj_call_count = 0
cur_best_score = 0 # 0 or np.inf
log_writer = open( '../log/lgb-hyperopt-log.txt', 'w' )


def objective(space):
    global obj_call_count, cur_best_score

    obj_call_count += 1

    print('\nLightGBM objective call #{} cur_best_score={:7.5f}'.format(obj_call_count,cur_best_score) )

    lgb_params = get_lgb_params(space)

    sorted_params = sorted(space.items(), key=lambda z: z[0])
    params_str = str.join(' ', ['{}={}'.format(k, v) for k, v in sorted_params])
    print('Params: {}'.format(params_str) )
    
    kf = KFold(n_splits=3, shuffle=True, random_state=0)
    out_of_fold = np.zeros(len(X_train))
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        D_train = lgb.Dataset(X_train.iloc[train_idx], label=Y_train[train_idx])
        D_val = lgb.Dataset(X_train.iloc[val_idx], label=Y_train[val_idx])
        # Train
        num_round = 10000
        clf = lgb.train(lgb_params,
                           D_train,
                           num_boost_round=num_round,
                           # metrics='mlogloss',
                           valid_sets=D_val,
                           # valid_names='val',
                           # fobj=None,
                           # feval=None,
                           # init_model=None,
                           # feature_name='auto',
                           # categorical_feature='auto',
                           early_stopping_rounds=200,
                           # evals_result=None,
                           verbose_eval=False,
                           # learning_rates=None,
                           # keep_training_booster=False,
                           # callbacks=None
                           )
        # predict
        nb_trees = clf.best_iteration
        val_loss = clf.best_score['valid_0']
        print('nb_trees={} val_loss={}'.format(nb_trees, val_loss))
        out_of_fold[val_idx] = clf.predict(X_train.iloc[val_idx], num_iteration=nb_trees)
        score = r2_score(out_of_fold, Y_train)

    print('val_r2_score={}'.format(score))

    log_writer.write('score={} Params:{} nb_trees={}\n'.format(score, params_str, nb_trees ))
    log_writer.flush()

    if score>cur_best_score:
        cur_best_score = score
        print(colorama.Fore.GREEN + 'NEW BEST SCORE={}'.format(cur_best_score) + colorama.Fore.RESET)
    return {'loss': -score, 'status': STATUS_OK}

# --------------------------------------------------------------------------------

space ={
        'num_leaves': hp.quniform ('num_leaves', 10, 100, 1),
        'min_data_in_leaf':  hp.quniform ('min_data_in_leaf', 10, 100, 1),
        'feature_fraction': hp.uniform('feature_fraction', 0.75, 1.0),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.75, 1.0),
        'learning_rate': hp.uniform('learning_rate', 0, 0.01),
#         'learning_rate': hp.loguniform('learning_rate', -5.0, -2.3),
        'min_sum_hessian_in_leaf': hp.loguniform('min_sum_hessian_in_leaf', 0, 2.3),
        'max_bin': hp.quniform ('max_bin', 88, 200, 1),
        'bagging_freq': hp.quniform ('bagging_freq', 1, 15, 1),
        'lambda_l1': hp.uniform('lambda_l1', 0, 10 ),
        'lambda_l2': hp.uniform('lambda_l2', 0, 10 ),
       }

trials = Trials()
best = hyperopt.fmin(fn=objective,
                     space=space,
                     algo=HYPEROPT_ALGO,
                     max_evals=N_HYPEROPT_PROBES,
                     trials=trials,
                     verbose=1)

print('-'*50)
print('The best params:')
print( best )
print('\n\n')