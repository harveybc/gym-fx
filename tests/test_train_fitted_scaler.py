import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from preprocessor_plugins.feature_window_preprocessor import Plugin


def frame():
    return pd.DataFrame(dict(DATE_TIME=pd.date_range('2020-01-01',periods=8,freq='h',tz='UTC'),
        price=np.arange(8,dtype=float)+100,constant=np.full(8,4.),cycle=np.linspace(-1,1,8)))


def fitted(tmp_path,passthrough=()):
    from preprocessor_plugins.train_fitted_scaler import fit_train,save
    data=frame();columns=['price','constant','cycle']
    body=fit_train(data,columns,split='train',start='2020-01-01T00:00:00Z',
        end='2021-01-01T00:00:00Z',source_sha256='a'*64,passthrough=passthrough)
    path=tmp_path/'scaler.json';pin=save(path,body)
    return data,body,path,pin


def config(path,pin):
    return dict(window_size=3,observation_stride_bars=1,require_full_observation_window=True,
        feature_columns=['price','constant','cycle'],feature_scaling='train_zscore',
        feature_scaler_file=str(path),feature_scaler_sha256=pin,
        feature_scaler_train_sha256='a'*64,feature_clip=0,include_price_window=False,
        include_agent_state=False)


def test_fixed_statistics_match_standard_scaler_and_constant_policy(tmp_path):
    from sklearn.preprocessing import StandardScaler
    data,body,path,pin=fitted(tmp_path)
    assert body['schema']=='gym_fx.train_fitted_zscore.v1' and body['fit']['rows']==8
    ref=StandardScaler().fit(data[['price','constant','cycle']])
    np.testing.assert_array_equal(body['mean'],ref.mean_)
    np.testing.assert_array_equal(body['scale'],ref.scale_)
    assert body['scale'][1]==1
    cfg=config(path,pin)
    actual=Plugin(cfg)._feature_window(data,8,cfg['feature_columns'],np.zeros(3,dtype=bool),cfg)
    np.testing.assert_array_equal(actual,ref.transform(data[['price','constant','cycle']]).astype(np.float32)[-3:])


def test_validation_does_not_refit_clip_or_modify_prices(tmp_path):
    data,body,path,pin=fitted(tmp_path);before=path.read_bytes()
    data['DATE_TIME']=pd.date_range('2024-01-01',periods=8,freq='h',tz='UTC')
    data['price']+=10000
    original=data.copy(deep=True);cfg=config(path,pin)
    plugin=Plugin(cfg)
    actual=plugin._feature_window(data,8,cfg['feature_columns'],np.zeros(3,dtype=bool),cfg)
    assert actual[:,0].min()>10
    pd.testing.assert_frame_equal(data,original)
    assert path.read_bytes()==before
    perturbed=data.copy();perturbed.loc[7,'price']=-10000
    a=Plugin(cfg)._feature_window(data,7,cfg['feature_columns'],np.zeros(3,dtype=bool),cfg)
    b=Plugin(cfg)._feature_window(perturbed,7,cfg['feature_columns'],np.zeros(3,dtype=bool),cfg)
    np.testing.assert_array_equal(a,b)
    assert not np.array_equal(actual,plugin._feature_window(perturbed,8,cfg['feature_columns'],np.zeros(3,dtype=bool),cfg))


def test_passthrough_preserves_native_cyclic_values(tmp_path):
    data,body,path,pin=fitted(tmp_path,['cycle']);cfg=config(path,pin)
    actual=Plugin(cfg)._feature_window(data,8,cfg['feature_columns'],np.zeros(3,dtype=bool),cfg)
    np.testing.assert_array_equal(actual[:,-1],data['cycle'].to_numpy(dtype=np.float32)[-3:])


@pytest.mark.parametrize('change',[dict(split='validation'),dict(split='test'),
    dict(end='2020-01-01T03:00:00Z'),dict(source_sha256='bad'),dict(passthrough=['missing'])])
def test_fit_rejects_holdout_boundaries_and_bad_identity(change):
    from preprocessor_plugins.train_fitted_scaler import fit_train
    args=dict(split='train',start='2020-01-01T00:00:00Z',end='2021-01-01T00:00:00Z',source_sha256='a'*64)
    args.update(change)
    with pytest.raises(ValueError):fit_train(frame(),['price','constant','cycle'],**args)


@pytest.mark.parametrize('corrupt',['nan','duplicate','order','naive','empty'])
def test_fit_rejects_invalid_data(corrupt):
    from preprocessor_plugins.train_fitted_scaler import fit_train
    data=frame()
    if corrupt=='nan':data.loc[3,'price']=np.nan
    if corrupt=='duplicate':data.loc[3,'DATE_TIME']=data.loc[2,'DATE_TIME']
    if corrupt=='order':data=data.iloc[::-1]
    if corrupt=='naive':data['DATE_TIME']=data['DATE_TIME'].dt.tz_localize(None)
    if corrupt=='empty':data=data.iloc[:0]
    with pytest.raises(ValueError):fit_train(data,['price','constant','cycle'],split='train',
        start='2020-01-01T00:00:00Z',end='2021-01-01T00:00:00Z',source_sha256='a'*64)


@pytest.mark.parametrize('corrupt',['bytes','columns','source','scale','clip_missing','nonfinite'])
def test_consumption_rejects_changed_artifacts_and_implicit_policies(tmp_path,corrupt):
    data,body,path,pin=fitted(tmp_path);cfg=config(path,pin)
    if corrupt=='bytes':path.write_bytes(path.read_bytes()+b' ')
    if corrupt=='columns':cfg['feature_columns']=list(reversed(cfg['feature_columns']))
    if corrupt=='source':cfg['feature_scaler_train_sha256']='b'*64
    if corrupt=='scale':
        body['scale'][0]=0;path.write_text(json.dumps(body));cfg['feature_scaler_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    if corrupt=='clip_missing':cfg.pop('feature_clip')
    if corrupt=='nonfinite':data.loc[7,'price']=np.inf
    with pytest.raises(ValueError):Plugin(cfg)._feature_window(data,8,cfg['feature_columns'],np.zeros(3,dtype=bool),cfg)


def test_save_reload_is_exact_and_cannot_overwrite_a_different_scaler(tmp_path):
    from preprocessor_plugins.train_fitted_scaler import save,load
    data,body,path,pin=fitted(tmp_path)
    assert save(path,body)==pin
    assert load(path,pin,['price','constant','cycle'],'a'*64)==body
    changed=dict(body,mean=[0,0,0])
    with pytest.raises(ValueError):save(path,changed)


def test_same_value_has_same_scale_at_different_origins(tmp_path):
    data,body,path,pin=fitted(tmp_path);cfg=config(path,pin)
    a=Plugin(cfg)._feature_window(data,7,cfg['feature_columns'],np.zeros(3,dtype=bool),cfg)
    b=Plugin(cfg)._feature_window(data,8,cfg['feature_columns'],np.zeros(3,dtype=bool),cfg)
    np.testing.assert_array_equal(a[1:],b[:-1])
