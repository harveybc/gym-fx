"""Frozen per-column TRAIN statistics; never fit while building observations."""
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd

SCHEMA='gym_fx.train_fitted_zscore.v1'


def _pin(value):
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None


def _stamp(value):
    stamp=pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError('FIXED_SCALER_AWARE_TIME_REQUIRED')
    return stamp.tz_convert('UTC')


def validate(body,columns,source_sha256):
    if (set(body)!={'schema','columns','passthrough','mean','scale','variance','fit'}
            or body['schema']!=SCHEMA or body['columns']!=list(columns)
            or not columns or len(set(columns))!=len(columns)
            or not isinstance(body['passthrough'],list)
            or len(set(body['passthrough']))!=len(body['passthrough'])
            or not set(body['passthrough']).issubset(columns)):
        raise ValueError('FIXED_SCALER_COLUMNS_OR_SCHEMA')
    for key in ('mean','scale','variance'):
        values=body[key]
        if (not isinstance(values,list) or len(values)!=len(columns)
                or any(type(v) not in (float,int) for v in values)
                or not np.isfinite(values).all()):
            raise ValueError('FIXED_SCALER_INVALID_STATISTICS')
    if any(v<=0 for v in body['scale']) or any(v<0 for v in body['variance']):
        raise ValueError('FIXED_SCALER_INVALID_SCALE')
    fit=body['fit']
    if (set(fit)!={'split','start','end','first','last','rows','source_sha256'}
            or fit['split']!='train' or not _pin(source_sha256)
            or fit['source_sha256']!=source_sha256
            or type(fit['rows']) is not int or fit['rows']<2
            or not _stamp(fit['start'])<=_stamp(fit['first'])<=_stamp(fit['last'])<_stamp(fit['end'])):
        raise ValueError('FIXED_SCALER_TRAIN_IDENTITY')
    return body


def fit_train(data,columns,*,split,start,end,source_sha256,passthrough=()):
    from sklearn.preprocessing import StandardScaler
    columns=list(columns);passthrough=list(passthrough)
    if (split!='train' or not _pin(source_sha256) or len(data)<2 or 'DATE_TIME' not in data
            or not columns or len(set(columns))!=len(columns)
            or not all(isinstance(c,str) and c for c in columns)
            or not set(columns).issubset(data.columns)
            or not set(passthrough).issubset(columns)
            or len(set(passthrough))!=len(passthrough)):
        raise ValueError('FIXED_SCALER_TRAIN_INPUT_REQUIRED')
    stamps=pd.DatetimeIndex([_stamp(t) for t in data['DATE_TIME']])
    if (not stamps.is_monotonic_increasing or not stamps.is_unique
            or stamps[0]<_stamp(start) or stamps[-1]>=_stamp(end)):
        raise ValueError('FIXED_SCALER_TRAIN_BOUNDARY')
    values=data[columns].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():raise ValueError('FIXED_SCALER_NONFINITE_TRAIN')
    scaler=StandardScaler().fit(values)
    body=dict(schema=SCHEMA,columns=columns,passthrough=passthrough,
        mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),variance=scaler.var_.tolist(),
        fit=dict(split='train',start=_stamp(start).isoformat(),end=_stamp(end).isoformat(),
            first=stamps[0].isoformat(),last=stamps[-1].isoformat(),rows=len(data),source_sha256=source_sha256))
    return validate(body,columns,source_sha256)


def save(path,body):
    validate(body,body['columns'],body['fit']['source_sha256'])
    payload=(json.dumps(body,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent,delete=False) as stream:
        temporary=Path(stream.name);stream.write(payload);stream.flush();os.fsync(stream.fileno())
    try:
        try:os.link(temporary,path)
        except FileExistsError:
            if path.read_bytes()!=payload:raise ValueError('FIXED_SCALER_IMMUTABLE_OUTPUT')
    finally:temporary.unlink()
    return hashlib.sha256(payload).hexdigest()


def load(path,pin,columns,source_sha256):
    payload=Path(path).read_bytes()
    if not _pin(pin) or hashlib.sha256(payload).hexdigest()!=pin:
        raise ValueError('FIXED_SCALER_DIGEST_MISMATCH')
    try:body=json.loads(payload)
    except (ValueError,TypeError) as exc:raise ValueError('FIXED_SCALER_INVALID_JSON') from exc
    try:return validate(body,columns,source_sha256)
    except (KeyError,TypeError) as exc:raise ValueError('FIXED_SCALER_INVALID_CONTRACT') from exc


def transform(values,body,*,clip):
    if type(clip) not in (int,float) or not np.isfinite(clip) or clip<0:
        raise ValueError('FIXED_SCALER_EXPLICIT_CLIP_REQUIRED')
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=2 or values.shape[1]!=len(body['columns']) or not np.isfinite(values).all():
        raise ValueError('FIXED_SCALER_NONFINITE_OR_SHAPE')
    scaled=(values-np.asarray(body['mean']))/np.asarray(body['scale'])
    for column in body['passthrough']:
        index=body['columns'].index(column);scaled[:,index]=values[:,index]
    if clip:np.clip(scaled,-clip,clip,out=scaled)
    with np.errstate(over='ignore',invalid='ignore'):scaled=scaled.astype(np.float32)
    if not np.isfinite(scaled).all():raise ValueError('FIXED_SCALER_NONFINITE_OUTPUT')
    return scaled
