#!/home/srvx11/lehre/users/a1254888/.conda/envs/a1254888/bin/python
# -*- coding: utf-8 -*-
"""
Author(s)
---------
Lukas Kugler (a1254888@univie.ac.at)

"""
from __future__ import division
import sys
import os
from shutil import copy
import datetime as dt
import numpy as np
import json, warnings
import pandas as pd

class Dataset(object):
    """This class handles the datasets.

    Attributes
    ----------
    self.obs_datasets : list
        list of observation datasets (= keys in self.df)

    self.fcst_datasets : list
        list of forecast datasets (= keys in self.df)

    self.df : pd.DataFrame
        Contains all data.
        Keys: are tuples of the form (dataset, parameter)
        Index: datetime
            one DataFrame row corresponds to one forecast,
            one observation may exist multiple times
            (f.e. lead time 12 h and 36 h with different analysisTime)

            index is unique if lagged_ensemble=True (older forecasts will
            be unstacked, i.e. horizontally appended)

            if there are ensemble members, they will also be unstacked.


    Workflow
    --------
        1) load datasets

        2) merge fcsts on valid_time
            (datasets = first column index)
            index may not be unique, because of multiple forecasts for the same
            valid_time.

        3) throw away short leadtime hours
           & subset on given lead time range

        4) add time-shifted variables as additional predictors
            = additional columns to existing `datasets`

        5) unstack on rows having the same 'valid_time'.
            this introduces another level on the column index, named 'fcst_id'

        6) join observations according to each row's 'valid_time'

    """
    def __init__(self, dataframes_obs, dataframes_fcst,
                 lagged_ensemble=False,
                 minmax_leadtime=(None, None),
                 steps_back=None):
        """
        Parameters
        ----------
        dataframes_obs : [(str, str, str),]
            list of tuple of (dataframe path, key, dataset nameää) of observations

            dataframe path : File path
            key : Identifier for the group in the store
            dataset name : Unique dataset descriptor

        dataframes_fcst : [(str, str, str),]
            list of tuple of (dataframe path, dataset name, key) of forecasts

            dataframe path : File path
            key : Identifier for the group in the store
            dataset name : Unique dataset descriptor

        lagged_ensemble : bool
            whether to use previous forecast runs as additional predictors

        minmax_leadtime : tuple of int
            contains minimum and maximum forecast hour (lead time) to consider
        """
        self.lagged_ensemble = lagged_ensemble
        self.minmax_leadtime = minmax_leadtime
        self.steps_back = steps_back
        #self.valid_time = None  # filled in later

        # configuration:
        name_level0 = 'dataset'  # f.e. ensemble member or different gridpoints
        name_level1 = 'forecast_nr'  # f.e. different forecast inits (lagged ensemble)
        name_level2 = 'parameter'  # f.e. T2M or DPT
        self.names = (name_level0, name_level1, name_level2)

        obs_dfs, fcst_dfs = self.load_datasets(dataframes_obs, dataframes_fcst)
        self.obs_datasets = obs_dfs.keys()
        self.fcst_datasets = fcst_dfs.keys()

        obs = self.prepare_obs(obs_dfs, steps_back=steps_back)
        fcsts = self.prepare_fcsts(fcst_dfs, steps_back=steps_back,
                                             lagged_ensemble=lagged_ensemble,
                                             minmax_leadtime=minmax_leadtime,)

        # merge fcsts and observations
        self.df = fcsts.join(obs, how='inner') #, on=(0, 0, 'valid_time'))
        self.df.index.name = 'valid_time'
        self.df.drop(labels=[(0, 0, 'valid_time')], axis=1, inplace=True)

    @staticmethod
    def load_datasets(dataframes_obs, dataframes_fcst):
        fcst_dfs = {}
        obs_dfs = {}

        print 'reading datasets ...'
        for (df_path, key, ds_name) in dataframes_obs:
             df = pd.read_hdf(df_path)
             obs_dfs[ds_name] = df[key] if isinstance(df, dict) else df

        for (df_path, key, ds_name) in dataframes_fcst:
            df = pd.read_hdf(df_path)
            fcst_dfs[ds_name] = df[key] if isinstance(df, dict) else df

        return obs_dfs, fcst_dfs

    @staticmethod
    def subset_fxh(df, minmax_leadtime):
        """Select a subset of forecast hours.
        Xy has multiindexed columns.

        Parameters
        ----------
        minmax_leadtime : (int, int)
            tuple of minimum and maximum lead time

        All values of one parameter are set to zero if the fxh
        corresponding to this forecast identifier is outside the range.
        """

        mmin, mmax = minmax_leadtime

        # mask axis0
        mask = np.ones(len(df), dtype=bool)
        datasets = df.keys().levels[0]

        for dataset in datasets:
            if dataset is not 'obs':
                fxh = df[(dataset, 'fxh')]
                a = 1 if mmin is None else (mmin <= fxh).values
                b = 1 if mmax is None else (fxh <= max).values
                mask_i = (a*b).astype(bool)
                mask *= mask_i

        return df.loc[mask,:]

    def prepare_obs(self, obs_dfs, steps_back=0):
        """
        obs_dfs : dict
            dict of (ds_name, pd.DataFrame) pairs
            df.index has to be datetime when the observation is valid

            Currently: len(obs_dfs.keys()) allowed is at most 1
        """
        if len(obs_dfs.keys()) > 1:
            raise NotImplementedError('This module currently only supports one data-'
                  'frame of observations: len(obs_dfs.keys()) > 1')

        # build observation DataFrame
        used_obs_df = obs_dfs.keys()[0]
        obs = obs_dfs[used_obs_df]
        obs['timeofday'] = obs.index.hour  # additional predictor probably

        if steps_back > 0:
            for parameter in obs.keys():
                if parameter in ('valid_time', 'init', 'fxh'):
                    continue
                # shift every predictor
                for i in range(1, steps_back+1):
                    # shift by all hours between 1 and steps_back
                    obs = self.add_shifted_predictor_onelevel(obs, parameter, shift_hours=i)

        # add hierarchical column
        keys = [('obs', 0, param) for param in obs.columns]
        obs.columns = pd.MultiIndex.from_tuples(keys, names=self.names)
        self.obs_keys = obs.keys()  # for use outside this module


        return obs

    def prepare_fcsts(self, fcst_dfs,
                      lagged_ensemble=None, steps_back=None,
                      minmax_leadtime=(None, None)):
        """
        fcst_dfs : dict
            dict of (ds_name, pd.DataFrame) pairs
        """

        # check input: `valid` is a mandatory field!
        for name, df in fcst_dfs.iteritems():
            if 'valid' not in df.keys():
                raise IOError('`valid` not in forecast DataFrame .keys(): ' \
                              +str(df.keys()))

        # merge fcst dataframes from different datasets and
        # add hierarchical columns for each dataset
        list_fcst_dfs = [df for name, df in fcst_dfs.iteritems()]
        ds_names = [name for name, df in fcst_dfs.iteritems()]

        # create index for merging, that contains valid time and forecast hour
        for df in list_fcst_dfs:
            df['mergeindex'] = df['valid'].map(str) + '-' + df['fxh'].map(str)
            df.set_index('mergeindex', inplace=True)

        # glue all forecasts together, with inner join on index=(valid & fxh)
        df_fcst = pd.concat(list_fcst_dfs, axis=1, join='inner', keys=ds_names)

        # replace mergeindex with valid time again
        some_dataset = df_fcst.keys().levels[0][0]
        df_fcst.set_index((some_dataset, 'valid'), inplace=True)
        df_fcst.index.name = 'valid'
        df_fcst.columns.names = [self.names[0], self.names[2]]
        print df_fcst.keys()

        # for use outside this module
        self.fcst_keys = [a for a in df_fcst.keys() if a not in self.obs_keys]

        def _ignore_spinup(df_fcst, hours_ignore=0):
            """Drop all rows with fxh less or equal to `hours_ignore`."""
            mask = np.ones(len(df_fcst), dtype=bool)
            for dataset in df_fcst.keys().levels[0]:
                mask *= (df_fcst[(dataset, 'fxh')] > hours_ignore).values
            print 'dropping', len(mask)-sum(mask), 'rows for spinup.'
            return df_fcst[mask]

        df_fcst = _ignore_spinup(df_fcst)

        select_leadtime_subset = any([a != None for a in minmax_leadtime])
        if select_leadtime_subset:
            df_fcst = self.subset_fxh(df_fcst, minmax_leadtime)

        # make index unique for `add_shifted_predictor
        df_fcst[(0, 'valid_time')] = df_fcst.index

        if steps_back > 0:
            for dataset, parameter in df_fcst.keys():
                if parameter in ('valid_time', 'init', 'fxh'):
                    continue
                # shift every predictor
                for i in range(1, steps_back+1):
                    # shift by all hours between 1 and steps_back
                    df_fcst = self.add_shifted_predictor(df_fcst,
                                                    (dataset, parameter),
                                                    shift_hours=i)
        #df_fcst.reset_index(inplace=True, drop=True)
        df_fcst.index = df_fcst[(0, 'valid_time')]
        #df_fcst.drop(labels=[(0, 'valid_time')], axis=1, inplace=True)

        def create_lagged_ensemble(df):
            """Make the index unique by putting the various forecast inits in one row.
            -> Instead of multiple rows for one valid_time use
            multiple columns for one valid_time
            """
            df.index.name = 'valid_time'
            df.reset_index(inplace=True)

            def tgrp(df):
                # if not drop valid_time, we get multiple columns 'valid_time'
                # but we only want one valid_time
                df = df.drop('valid_time', axis=1)
                return df.reset_index(drop=True)

            df = df.groupby('valid_time').apply(tgrp).unstack()
            df.columns = pd.MultiIndex.from_tuples(
                                            [(a[0], a[2], a[1]) for a in df.columns],
                                            names=self.names)
            df = df[sorted(df.columns)]
            return df

        if lagged_ensemble:
            print 'putting lagged ensemble forecasts side by side - unique indexing'
            df_fcst = create_lagged_ensemble(df_fcst)
        else:
            # add dummy second column level with entry "0"
            # such that the keys are the same as with `lagged_ensemble`
            df_fcst.columns = pd.MultiIndex.from_tuples(
                                            [(a[0], 0, a[1]) for a in df_fcst.columns],
                                            names=self.names)

        if self.lagged_ensemble:
            # self.df.columns-index is tuple, convert it to MultiIndex again.
            tuples = list(df_fcst.columns)
            df_fcst.columns = pd.MultiIndex.from_tuples(tuples, names=self.names)

        def check_validity(df):
            """Random sample test the validity of the join operations.

            To check the validity of the join,
            see if all rows with the same datetime have the same obs.
            Because there can not be two different observations on the same time.
            """
            series_random = df.iloc[int(np.random.rand()*len(df))]
            c = ('obs', 'T')

            df_onetime =  df.loc[series_random.name]
            s = df_onetime
            print 'check_validity', s[c], s
            values = s[c] - s[c].mean()
            assert all(v == 0 for v in values)

        #check_validity(self.df)
        if not df_fcst.index.is_unique:
            print ('index not unique')
            #df_fcst.reset_index(inplace=True, drop=True)

        return df_fcst

    def build_predictors_list(self, parameters_to_skip=['valid', 'init', 'fxh'],
                              obs_params_to_use=['timeofday']):
        """Build the list of predictors from keys in self.df.fcst.

        Parameters
        ----------
        datasets_to_use : [str,]
            top level keys in self.df

        parameters_to_skip : [str,]
            not use these variables for prediction
        """
        predictors = []
        if not parameters_to_skip: parameters_to_skip = []

        for (dataset, id, param) in self.df.keys():
            if dataset == 'obs':
                if param in obs_params_to_use:
                    predictors.append((dataset, 0, param))
            else:
                if param not in parameters_to_skip:
                    predictors.append((dataset, id, param))
        return predictors

    def create_binary_parameter_from_parameters(self, criteria):
        """Create a binary predictand/target from criteria on data columns.

        Parameters
        ----------
        criteria : {str: (float, float),}
            criteria to create a new binary parameter from existing ones

            i.e. the criterium (T_obs < 0) & ((T_obs - Td_obs) < 1.)
            on the timeseries of T_obs and Td_obs
            -> criteria = {'T_obs': (None, 0), 'Spread_obs': (None, 1.)}
            keys have to be columns of df
        """
        mask = np.ones(len(self.df))
        for key, minmax in criteria.iteritems():
            a = 1 if minmax[0] == None else (minmax[0] <= self.df[key])
            b = 1 if minmax[1] == None else (self.df[key] <= minmax[1])
            mask_i = a*b
            mask *= mask_i
        return mask.astype(int)  # return the predictand 0/1

    def create_time_aggregated_parameter(self, column_to_agg, agg_func, agg_timedelta):
        """Create a parameter by aggregating a data column over a number
    	of elements.

        Parameters
        ----------
        	column_to_agg : str
        	    how the column to aggregate on is called in self.df

        	agg_func : attribute of pandas.core.window.Rolling
        	    how to aggregate, f.e. 'mean', 'max', 'sum'

        	agg_timedelta : datetime.timedelta
        	    over which period to aggregate over
        """
        return getattr(self.df[column_to_agg].rolling(agg_timedelta, min_periods=1),
                                       agg_func)()

    @staticmethod
    def add_shifted_predictor(df, orig_key, shift_hours=1):
        """Add the predicted value of a parameter ´shift_hours´ hours ago
        as a new parameter/predictor.

        join on: same

        Parameters
        ----------
            orig_fcst : pd.DataFrame
                Dataframe before adding shifted parameters

            orig_key : tuple or str
                key of parameter to shift


        We do this by shifting the valid time by one hour,
        and merging the df with the original dataframe with matching
        valid and init time.
        """
        assert isinstance(shift_hours, int)
        assert shift_hours != 0

        dataset, param = orig_key
        new_key = (dataset, param+'-'+str(shift_hours))

        shifted_time = df[(0, 'valid_time')] + dt.timedelta(hours=shift_hours)
        fxh = df[(dataset, 'fxh')].values+shift_hours

        fcst_shift = pd.DataFrame(data={new_key: df[orig_key].values,
                                        (dataset, 'fxh'): fxh,
                                        (0, 'valid_time'): shifted_time})

        df = pd.merge(df, fcst_shift, how='left',
                      on=[(0, 'valid_time'), (dataset, 'fxh')])

        df.index = df[(0, 'valid_time')]
        return df

    @staticmethod
    def add_shifted_predictor_onelevel(df, orig_key, shift_hours=1):
        """Add the predicted value of a parameter ´shift_hours´ hours ago
        as a new parameter/predictor.

        join on: same

        Parameters
        ----------
            orig_fcst : pd.DataFrame
                Dataframe before adding shifted parameters

            orig_key : tuple or str
                key of parameter to shift


        We do this by shifting the valid time by one hour,
        and merging the df with the original dataframe with matching
        valid and init time.
        """
        assert isinstance(shift_hours, int)
        assert shift_hours != 0

        param = orig_key
        new_key = param+'-'+str(shift_hours)

        shifted_time = df.index + dt.timedelta(hours=shift_hours)

        fcst_shift = pd.DataFrame(index=shifted_time,
                                  data={new_key: df[orig_key].values})

        df = df.join(fcst_shift, how='left')
        return df
