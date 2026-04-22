#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plugin_loader.py

Module for loading plugins using the importlib.metadata entry points API updated for Python 3.12.
Provides functions to load a specific plugin and retrieve its parameters.
"""

from importlib.metadata import entry_points, EntryPoint

def load_plugin(plugin_group: str, plugin_name: str):
    """
    Load a plugin class from a specified entry point group using its name.
    
    This function uses the updated importlib.metadata API for Python 3.12 by filtering 
    entry points with the select() method.

    Args:
        plugin_group (str): The entry point group from which to load the plugin.
        plugin_name (str): The name of the plugin to load.

    Returns:
        tuple: A tuple containing the plugin class and a list of required parameter keys 
               extracted from the plugin's plugin_params attribute.

    Raises:
        ImportError: If the plugin is not found in the specified group.
        Exception: For any other errors during the plugin loading process.
    """
    print(f"Attempting to load plugin: {plugin_name} from group: {plugin_group}")
    try:
        # Filter entry points for the specified group using the new .select() method.
        group_entries = entry_points().select(group=plugin_group)
        # Find the entry point that matches the plugin name.
        entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
        # Load the plugin class using the entry point's load method.
        plugin_class = entry_point.load()
        # Extract the keys from the plugin's plugin_params attribute as required parameters.
        required_params = list(plugin_class.plugin_params.keys())
        print(f"Successfully loaded plugin: {plugin_name} with params: {plugin_class.plugin_params}")
        return plugin_class, required_params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to load plugin {plugin_name} from group {plugin_group}, Error: {e}")
        raise

def get_plugin_params(plugin_group: str, plugin_name: str):
    """
    Retrieve the plugin parameters from a specified entry point group using the plugin name.
    
    This function loads the plugin class using the updated importlib.metadata API and returns 
    its plugin_params attribute.

    Args:
        plugin_group (str): The entry point group from which to retrieve the plugin.
        plugin_name (str): The name of the plugin.

    Returns:
        dict: A dictionary representing the plugin parameters (plugin_params).

    Raises:
        ImportError: If the plugin is not found in the specified group.
        ImportError: For any errors encountered while retrieving the plugin parameters.
    """
    print(f"Getting plugin parameters for: {plugin_name} from group: {plugin_group}")
    try:
        # Filter entry points for the specified group using the new .select() method.
        group_entries = entry_points().select(group=plugin_group)
        # Find the entry point that matches the plugin name.
        entry_point = next(ep for ep in group_entries if ep.name == plugin_name)
        # Load the plugin class using the entry point's load method.
        plugin_class = entry_point.load()
        print(f"Retrieved plugin params: {plugin_class.plugin_params}")
        return plugin_class.plugin_params
    except StopIteration:
        print(f"Failed to find plugin {plugin_name} in group {plugin_group}")
        raise ImportError(f"Plugin {plugin_name} not found in group {plugin_group}.")
    except Exception as e:
        print(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
        raise ImportError(f"Failed to get plugin params for {plugin_name} from group {plugin_group}, Error: {e}")
