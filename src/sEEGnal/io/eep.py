# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 03:24:18 2022

@author: Ricardo Bruña

Based on the description of the EEP 3.x file format in:
* cnt_riff.txt by Rainer Nowagk & Maren Grigutsch.

"""

import re
import pandas
import numpy

import sEEGnal.io.riff as riff
import sEEGnal.io.raweep as raweep
import sEEGnal.tools.mne_tools as mne_tools


"""
Code for reading and parsing the EEP header.
"""

# Function to read the header of the EEProbe file.
def read_info ( filename ):
    
    # Function to parse the raw events intro comprehensible data.
    def parse_events ( info ):
        
        
        # If no raw event definition, exists.
        if 'rawevents' not in info:
            return info
        
        
        # Gets the raw event definition.
        rawevents = info [ 'rawevents' ]
        
        
        # Initializes the list of events.
        events = []
        
        
        # Goes through each event.
        for rawevent in rawevents:
            
            # Gets the event information.
            etype    = rawevent [ 'event' ] [ 'uname' ]
            etime    = rawevent [ 'event' ] [ 'time' ]
            evalue   = rawevent [ 'event' ] [ 'state' ]
            elength  = rawevent [ 'event' ] [ 'duration' ]
            edesc    = rawevent [ 'description' ]
            erawdata = rawevent [ 'event' ] [ 'epoch_desc' ]
            
            # Parses the epoch events.
            if rawevent [ 'description' ] == 'Epoch Event':
                evalue   = erawdata [0] [ 'data']
            
            
            # Estimates the event onset respect to the segment onset.
            eonset   = etime - numpy.array ( info [ 'segments' ].start_time )
            esegment = numpy.where ( eonset > -1 ) [0] [-1]
            eonset   = min ( eonset [ eonset > -1 ] )
            esample  = numpy.floor ( eonset * info [ 'sample_rate' ] )
            sonset   = info [ 'segments' ].start_sample [ esegment ] / info [ 'sample_rate' ]
            ssample  = info [ 'segments' ].start_sample [ esegment ]
            
            
            # Builds the event structure.
            event = {
                'type':        etype,
                'onset':       eonset + sonset,
                'sample':      int ( esample + ssample ),
                'value':       evalue,
                'duration':    elength,
                'description': edesc,
                'rawdata':     erawdata,
                'timestamp':   etime }
            
            # Stores the event information.
            events.append ( event )
        
        
        # Adds the information about the segments, if any.
        for _, segment in info [ 'segments' ].iterrows ():
            
            # Gets the segment information.
            etype    = 'Segment'
            etime    = segment [ 'start_time' ]
            evalue   = 0
            elength  = segment [ 'sample_count' ] / info [ 'sample_rate' ]
            edesc    = 'New segment'
            eonset   = segment [ 'start_sample' ] / info [ 'sample_rate' ]
            esample  = segment [ 'start_sample' ]
            
            # Builds the event structure.
            event = {
                'type':        etype,
                'onset':       eonset,
                'sample':      esample,
                'value':       evalue,
                'duration':    elength,
                'description': edesc,
                'rawdata':     None,
                'timestamp':   etime }
            
            # Stores the segment information.
            events.append ( event )
        
        
        # Sorts the events by onset.
        onsets = [ event [ 'onset' ] for event in events ]
        order  = numpy.argsort ( onsets )
        events = [ events [ index ] for index in order ]
        
        
        # Adds the events to the file information.
        info [ 'events' ] = events
        
        # Removes the raw events.
        del info [ 'rawevents' ]
        
        # Returns the updated information.
        return info
    
    
    
    # Function to parse impedance events into comprehensible data.
    def parse_imp ( info ):
        
        
        # If no event definition, exists.
        if 'events' not in info:
            return info
        
        
        # Lists the channel labels.
        labels   = info [ 'channels' ] [ 'label' ]
        
        # Initializes the list of impedance measurements.
        imps     = []
        
        # Goes through each event.
        for event in info [ 'events' ]:
            
            # If the event is an impedance measurement, takes the value.
            if event [ 'description' ] == 'Impedance':
                
                # Gets the impedance information.
                imp_vals = event [ 'rawdata' ] [0] [ 'data' ]
                imp_unit = event [ 'rawdata' ] [0] [ 'unit' ]
                imp_time = event [ 'timestamp' ]
                
                # Generates a nested dictionary for the impedances.
                imp = {
                    'time':        imp_time,
                    'unit':        imp_unit,
                    'measurement': dict ( zip ( labels, imp_vals ) ) }
                
                # Stores the measurement.
                imps.append ( imp )
        
        
        # Adds the impedances to the information.
        info [ 'impedances' ] = imps
        
        # Returns the updated information.
        return info
    
    
    
    # Function to parse video events into comprehensible data.
    def parse_video ( info ):
        
        
        # If no event definition, exists.
        if 'events' not in info:
            return info
        
        
        # Gets the raw event definition.
        events   = info [ 'events' ]
        
        # Initializes the list of video files.
        videos   = []
        
        
        # Goes through each event.
        for event in events:
            
            # If the event is a video, takes the file name.
            if event [ 'description' ].startswith ( 'Video' ):
                
                # Gets the video information.
                video_time = event [ 'onset' ]
                video_len  = event [ 'duration' ]
                video_desc = event [ 'description' ]
                video_file = event [ 'rawdata' ] [1] [ 'data' ]
                
                # Builds the video structure.
                video = {
                    'onset':       video_time,
                    'length':      video_len,
                    'description': video_desc,
                    'filename':    video_file }
                
                # Stores the video information.
                videos.append ( video )
            
            
        # Adds the list of videos to the information.
        info [ 'videos' ] = videos
        
        # Returns the updated information.
        return info
    
    
    # Reads the RIFF file tree, if required.
    rifftree = riff.read_file ( filename )
    
    
    # Initializes the information dictionary.
    info     = {}
    
    
    # Gets the EEP header.
    subtree  = riff.get_subtree ( rifftree, [ 'eeph' ] )
    dummy    = bytes ( subtree [ 'data' ] ).decode ()
    
    # Looks for the file version.
    pattern = r'\[File Version]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len ( hits ) > 0:
        info [ 'file_version' ] = hits [0]
    
    # Looks for the sampling rate.
    pattern = r'\[Sampling Rate]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len(hits) > 0:
        info['sample_rate'] = int(float(hits[0]))
    
    # Looks for the number of samples.
    pattern = r'\[Samples]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len(hits) > 0:
        info['sample_count'] = int(float(hits[0]))
    
    # Looks for the number of channels.
    pattern = r'\[Channels]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len(hits) > 0:
        info['channel_count'] = int(float(hits[0]))
    
    # Looks for the channel information.
    pattern = r'\[Basic Channel Data]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    assert len ( hits ) > 0, 'No channel definition. Cannot continue.'
    
    # Checks for the (mandatory) channel information header.
    chaninfo = hits [0].strip ()
    hits     = re.split ( ';label.*calibration factor(.*)', chaninfo, 0, re.S )
    assert len ( hits ) > 2, 'The channel information is not correct. Cannot continue.'
    
    # Parses the channel definition.
    chaninfo = hits [1].strip ()
    chaninfo = [ x.split () for x in chaninfo.splitlines () ];
    chaninfo = pandas.DataFrame ( chaninfo );
    
    # Combines the calibration factors.
    chaninfo [1] = chaninfo [1].astype ( float ) * chaninfo [2].astype ( float )
    chaninfo = chaninfo [ [ 0, 1, 3, 4 ] ]
    
    # Adds the labels.
    chaninfo.columns = [ 'label', 'calibration', 'unit', 'reference' ]
    
    # Stores the channel information.
    info [ 'channels' ] = chaninfo
    
    
    # Gets the EEP header.
    subtree  = riff.get_subtree ( rifftree, [ 'info' ] )
    dummy    = bytes ( subtree [ 'data' ] ).decode ()
    
    # Looks for the acquistion date and fraction.
    pattern = r'\[StartDate]\n([^\[]+)\n\[StartFraction]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len ( hits ) > 0:
        acqdate  = float ( hits [0][0] )
        acqfrac  = float ( hits [0][1] )
        
        # Converts the acquisition date into POSIX time format.
        info [ 'acquisition_time' ] = acqdate * ( 60 * 60 * 24 ) - 2209161600 + acqfrac
    
    # Looks for the file version.
    pattern = r'\[File Version]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len ( hits ) > 0:
        info [ 'file_version' ] = hits [0]
    
    # Looks for the software identification.
    pattern = r'\[MachineMake]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len ( hits ) > 0:
        info [ 'software_id' ] = hits [0]
    
    # Looks for the amplifier identification.
    pattern = r'\[MachineModel]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len ( hits ) > 0:
        info [ 'hardware_id' ] = hits [0]
    
    # Looks for the subject name.
    pattern = r'\[SubjectName]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len ( hits ) > 0:
        info [ 'subject_name' ] = hits [0]
    
    # Looks for the subject birth date.
    pattern = r'\[SubjectDateOfBirth]\n([^\[]+)\n'
    hits = re.findall(pattern, dummy)
    if len ( hits ) > 0:
        info [ 'subject_birth' ] = hits [0]
    
    
    # # Gets the event definition.
    # subtree  = riff.get_subtree ( rifftree, [ 'evt' ] )
    # info [ 'rawevent' ] = subtree [ 'data' ]
    
    
    
    # # Other fields not used.
    # subtree  = riff.get_subtree ( rifftree, [ 'tfh' ] )
    
    
    
    # Tries to read the sidecar segment file.
    segfile  = re.sub ( '.cnt$', '.seg', filename )
    segments = read_seg ( segfile )
    
    
    # Creates a DataFrame for the first segment.
    segment1 = pandas.DataFrame ()
    segment1 [ 'identifier'   ] = [ 0 ]
    segment1 [ 'start_time'   ] = [ info [ 'acquisition_time' ] ]
    segment1 [ 'sample_count' ] = [ info [ 'sample_count' ] - sum ( segments.sample_count ) ]
    
    # Concatenates both DataFrames.
    segments = pandas.concat ( [ segment1, segments ], ignore_index = True )
    
    # Gets the start sample for each segment.
    segstart = info [ 'sample_count' ] - segments.sample_count [ ::-1 ].cumsum () [ ::-1 ];
    segments [ 'start_sample' ] = segstart
    
    
    # Stores the segment definition.
    info [ 'segments' ] = segments
    
    
    
    # Tries to read the sidecar event file.
    evtfile  = re.sub ( '.cnt$', '.evt', filename )
    info [ 'rawevents' ] = read_evt ( evtfile )
    
    
    # Parses the events, impedances, and video files.
    info = parse_events ( info )
    info = parse_imp ( info )
    info = parse_video ( info )
    
    # Removes the raw event data.
    for event in info [ 'events' ]:
        del event [ 'rawdata' ]
    
    
    # Returns the information dictionary.
    return info




"""
Code for reading the sidecar events file.

Based on libeep 3.3.177 functions on:
* libcnt/evt.c
* libcnt/evt.h
* v4/eep.c
* v4/eep.h

"""

# Main function to read *.evt (event) files.
def read_evt ( filename ):
    
    # Function to read an event library from an *.evt file.
    def read_library ( fid ):
        
        
        # Reads the library name.
        name      = read_string ( fid )
        
        # Reads the number of events.
        nevent    = numpy.fromfile ( fid, 'uint32', 1 ) [0]
        
        # Initializes the list of events.
        events    = []
        
        # Goes through each event.
        for eindex in range ( nevent ):
            
            # Reads the event class.
            cname    = read_class ( fid )
            
            # Reads the event using the class-specific function.
            if cname == 'class dcEpochEvent_c':
                event = read_epoch    ( fid )
                
            elif cname == 'class dcEventMarker_c':
                event = read_marker   ( fid )
                
            elif cname == 'class dcArtefactEvent_c':
                event = read_artefact ( fid )
                
            elif cname == 'class dcSpikeEvent_c':
                event = read_spike    ( fid )
                
            elif cname == 'class dcSeizureEvent_c':
                event = read_seizure  ( fid )
                
            elif cname == 'class dcSleepEvent_c':
                event = read_sleep    ( fid )
                
            elif cname == 'class dcRPeakEvent_c':
                event = read_rpeak    ( fid )
                
            else:
                raise ValueError ( 'Unknown event class.' )
        
            # Stores the event.
            events = events + [ event ];
        
        
        # Sets the ouput.
        library = {
            'name':    name,
            'entries': events }
        
        # Returns the library.
        return library
    
    
    
    # Function to read event entries of type "epoch".
    def read_epoch    ( fid ):
        
        
        # Reads the event.
        event    = read_event ( fid )
        
        
        # if version < 33
        #   numpy.fromfile ( fid, 'int32', 1 ) [0]
        # end
        
        
        # Sets the ouput.
        marker      = {
        'class':          'marker',
        'event':          event,
        'chaninfo':       [],
        'description':    event [ 'uname' ],
        'show_amplitude': 0,
        'show_duration':  0 }
        
        # Returns the marker dictionary.
        return marker
        
    
    
    # Function to read event entries of type "marker".
    def read_marker   ( fid ):
        
        
        # Reads the event.
        event    = read_event ( fid )
        
        # Reads the channel information.
        chaninfo = read_chaninfo ( fid )
        
        # Reads the marker description.
        desc     = read_string ( fid )
        
        
        # if version >= 35
        #   if version >= 103
        show_amplitude = numpy.fromfile ( fid, 'int32', 1 ) [0]
        #   else
        #     show_amplitude = fread ( fid, 1, '*int8' );
        #   end
        show_duration = numpy.fromfile ( fid, 'int8', 1 ) [0]
        # end
        
        
        # Sets the ouput.
        marker      = {
        'class':          'marker',
        'event':          event,
        'chaninfo':       chaninfo,
        'description':    desc,
        'show_amplitude': show_amplitude,
        'show_duration':  show_duration }
        
        # Returns the marker dictionary.
        return marker
        
    
    
    # Funtions not yet defeloped to read other type of entries.
    def read_artefact ( fid ): raise NotImplementedError ( 'Not yet coded.' )
    def read_spike    ( fid ): raise NotImplementedError ( 'Not yet coded.' )
    def read_seizure  ( fid ): raise NotImplementedError ( 'Not yet coded.' )
    def read_sleep    ( fid ): raise NotImplementedError ( 'Not yet coded.' )
    def read_rpeak    ( fid ): raise NotImplementedError ( 'Not yet coded.' )
    
    
    
    # Function to read the basic event information.
    def read_event ( fid ):
        
        
        # Reads the event identifier.
        ident = numpy.fromfile ( fid, 'int32', 1 ) [0]
        
        # Reads the event GUID.
        numpy.fromfile ( fid, 'uint32', 1 )
        numpy.fromfile ( fid, 'uint16', 1 )
        numpy.fromfile ( fid, 'uint16', 1 )
        numpy.fromfile ( fid, 'uint8', 8 )
        
        
        # Reads the event class.
        cname    = read_class ( fid )
        
        # Reads the event name.
        uname    = read_string ( fid )
        ename    = read_string ( fid )
        
        # Reads the event type and state.
        etype    = numpy.fromfile ( fid, 'int32', 1 ) [0]
        state    = numpy.fromfile ( fid, 'int32', 1 ) [0]
        
        # Reads the original tag.
        original = numpy.fromfile ( fid, 'int8', 1 ) [0]
        
        # Reads the event duration.
        duration = numpy.fromfile ( fid, 'float64', 1 ) [0]
        offset   = numpy.fromfile ( fid, 'float64', 1 ) [0]
        
        # Reads the timestamp.
        time     = numpy.fromfile ( fid, 'float64', 2 )
        time     = time [0] * ( 60 * 60 * 24 ) - 2209161600 + time [1]
    
        # Reads the epoch descriptors.
        epoch    = read_epoch_descriptors ( fid )
        
        
        # Sets the ouput.
        event     = {
            'id':         ident,
            'class':      cname,
            'uname':      uname,
            'name':       ename,
            'type':       etype,
            'state':      state,
            'original':   original,
            'duration':   duration,
            'offset':     offset,
            'time':       time,
            'epoch_desc': epoch }
        
        # Returns the event.
        return event
    
    
    
    # Function to read the event descriptor(s).
    def read_epoch_descriptors ( fid ):
        
        
        # Gets the number of epoch descriptors.
        ndesc  = numpy.fromfile ( fid, 'int32', 1 ) [0]
        
        # Initializes the list of descriptors.
        descs  = []
        
        # Goes through each descriptor.
        for dindex in range ( ndesc ):
            
            # Gets the name of the descriptor.
            dname    = read_string ( fid )
            
            # Reads the data.
            ddata    = read_data ( fid )
            
            # Reads the descriptor unit.
            dunit    = read_string ( fid )
            
            # Stores the output.
            desc     = {
                'name': dname,
                'data': ddata,
                'unit': dunit }
            
            descs    = descs + [ desc ];
        
        
        # Returns the descriptors.
        return descs
    
    
    
    # Function to read the channel affected by the event.
    def read_chaninfo ( fid ):
        
        # Gets the active channel and the reference.
        active  = read_string ( fid )
        ref     = read_string ( fid )
        
        
        # Prepares the output.
        chaninfo = {
            'active': active,
            'ref':    ref }
        
        # Returns the channel information.
        return chaninfo
    
    
    
    # Helper function to read the entry class.
    def read_class ( fid ):
        
        # Reads the class tag.
        tag   = numpy.fromfile ( fid, 'int32', 1 ) [0]
        
        # If no tag, exits.
        if tag == 0:
            return ''
        
        assert tag == -1, 'Unknown class tag.'
        
        # Reads the class name.
        cname = read_string ( fid )
        
        # Returns the class name.
        return cname
    
    
    
    # Helper function to read a data piece.
    def read_data ( fid ):
        
        
        # Gets the type of data.
        datatype = numpy.fromfile ( fid, 'int16', 1 ) [0]
        
        # Reads the data.
        if datatype == 0:
            pass
        
        elif datatype == 1:
            pass
        
        elif datatype == 2:
            data     = numpy.fromfile ( fid, 'int16', 1 ) [0]
        
        elif datatype == 3:
            data     = numpy.fromfile ( fid, 'int32', 1 ) [0]
        
        elif datatype == 4:
            data     = numpy.fromfile ( fid, 'float32', 1 ) [0]
        
        elif datatype == 5:
            data     = numpy.fromfile ( fid, 'float64', 1 ) [0]
        
        elif datatype == 8:
            data     = read_unicode ( fid )
        
        elif datatype == 11:
            raise ValueError ( 'Data type not supported.' )
        
        elif datatype == 2 ** 9:
            data     = read_array ( fid )
        
        elif datatype == 2 ** 10:
            pass
        
        elif datatype & ( 2 ** 13 | 2 ** 14 ):
            data     = read_array ( fid )
        
        else:
            raise ValueError ( 'Data type not supported.' )
        
        
        # Returns the read data.
        return data
    
    
    
    # Helper function to read a data array.
    def read_array ( fid ):
        
        # Gets the type of data.
        datatype = numpy.fromfile ( fid, 'int16', 1 ) [0]
        
        # Reads a dummy element.
        if datatype == 4:
            numpy.fromfile ( fid, 'float32', 1 )
        
        else:
            raise ValueError ( 'Data type not supported.' )
        
        
        # Gets the length of the array.
        datalen  = numpy.fromfile ( fid, 'uint32', 1 ) [0]
        
        # Reads the data.
        if datatype == 4:
            data     = numpy.fromfile ( fid, 'float32', datalen )
        
        else:
            raise ValueError ( 'Data type not supported.' )
        
        
        # Returns the array.
        return data
    
    
    
    # Helper function to read Unicode (utf-16) strings.
    def read_string ( fid ):
        
        # Gets the length of the text.
        length = numpy.fromfile ( fid, 'uint8', 1 ) [0]
        assert length < 255, 'Text too long'
        
        # Reads the text.
        string = numpy.fromfile ( fid, 'uint8', length )
        string = bytes ( string ).decode ()
        
        # Returns the string.
        return string
    
    
    
    # Helper function to read Unicode (utf-16) strings.
    def read_unicode ( fid ):
        
        # Gets the length of the text.
        length = numpy.fromfile ( fid, 'int32', 1 ) [0]
        assert length < 255, 'Text too long'
        
        # Reads the text.
        string = numpy.fromfile ( fid, 'uint8', length )
        string = bytes ( string ).decode ( 'utf16' )
        
        # Returns the string.
        return string
    
    
    
    # Opens the file to read.
    with open ( filename, 'rb' ) as fid:
        
        # Reads the event header.
        time     = numpy.fromfile ( fid, 'uint32', 3 )
        version  = numpy.fromfile ( fid, 'int32', 1 ) [0]
        compress = numpy.fromfile ( fid, 'int32', 1 ) [0]
        encrpyt  = numpy.fromfile ( fid, 'int32', 1 ) [0]
        
        # Checks the file header.
        assert compress == 0, 'This function only works with noncompressed event files.'
        assert encrpyt == 0, 'This function only works with nonencrypted event files.'
        
        
        # Gets the main class.
        cname    = read_class ( fid )
        assert cname == 'class dcEventsLibrary_c', 'Bad file.'
        
        # Reads the event library.
        library  = read_library ( fid )
        
        # Adds the header.
        library [ 'time ']     = time
        library [ 'version ']  = version
        library [ 'compress '] = compress
        library [ 'encrpyt ']  = encrpyt
    
    
    # Returns only the contents of the library.
    event    = library [ 'entries' ]
    return event




"""
Code for reading the sidecard segments file.
"""

# Function to read the EEProbe segment (*.seg) file.
def read_seg ( filename ):
    
    
    # Opens the file to read.
    try:
        with open ( filename, 'rb' ) as fid:
        
            # Gets the total file length.
            fid.seek ( 0, 2 )
            flen = fid.tell ()
            fid.seek ( 0, 0 )
            
            # Reads the data.
            rawseg = numpy.fromfile ( fid, 'uint8', flen )
            rawseg = bytes ( rawseg ).decode ()
    
    # If no file generates a dummy one.
    except FileNotFoundError:
        rawseg = 'NumberSegments=1\nNaN NaN NaN'
    
    
    # Looks for the segment definition.
    pattern = r'NumberSegments=(.*)\n(.*)'
    hits = re.findall(pattern, rawseg)
    assert len ( hits ) > 0, 'Segment file is corrupted.'
    
    # Gets the number of segments and the data.
    nseg     = int ( hits [0][0] )
    
    # Parses the segment definition.
    segdata  = hits [0][1].strip ()
    segdata  = [ x.split () for x in segdata.splitlines () ]
    segdata  = pandas.DataFrame ( segdata, dtype = float )
    segdata  = segdata [ :nseg - 1 ]
    
    
    # Stores the segment information.
    segments = pandas.DataFrame ()
    segments [ 'identifier'   ] = range ( 1, nseg )
    segments [ 'start_time'   ] = segdata [0] * ( 60 * 60 * 24 ) - 2209161600 + segdata [1]
    segments [ 'sample_count' ] = segdata [2].astype ( int )
    
    
    
    # Returns the segments list.
    return segments



"""
Code for reading the data from the compressed data stream.
"""

# Function to parse the raw data (raw3 field) of the EEProbe file.
def read_rawdata ( filename ):
    
    
    # Reads the RIFF file tree, if required.
    rifftree = riff.read_file ( filename )
    
    
    # Gets the raw data epoch information.
    subtree  = riff.get_subtree ( rifftree, [ 'raw3', 'ep' ] )
    dummy    = subtree [ 'data' ]
    dummy    = dummy.reshape ( [ -1, 8 ] )
    dummy    = [ int.from_bytes ( item, 'little' ) for item in dummy ]
    
    # Gets the number of epochs and the samples per epoch.
    sepoch   = dummy [0]
    epochs   = dummy [ 1: ]
    nepoch   = len ( epochs )
    
    
    # Gets the raw data channel information.
    subtree  = riff.get_subtree ( rifftree, [ 'raw3', 'chan' ] )
    dummy    = subtree [ 'data' ]
    dummy    = dummy.reshape ( [ -1, 2 ] )
    dummy    = [ int.from_bytes ( item, 'little' ) for item in dummy ]
    
    # Gets the raw data channel order.
    chorder  = dummy
    
    # Gets the raw data itself.
    subtree  = riff.get_subtree ( rifftree, [ 'raw3', 'data' ] )
    data    = subtree [ 'data' ]
    
    
    # Prepares the raw data information.
    rawdata = {
        'epoch_count':   nepoch,
        'epoch_length':  sepoch,
        'epoch_start':   epochs,
        'channel_order': chorder,
        'data':          data }
    
    # Returns the raw data dictionary.
    return rawdata



# Function to read 
def read_data ( filename, info = None ):
    
    
    # Gets the EEProbe header and raw data.
    info     = read_info ( filename )
    rawdata  = read_rawdata ( filename )
    
    
    # Gets the information from the system-specific header.
    nchan    = info [ 'channel_count' ]
    nsamp    = info [ 'sample_count' ]
    chcalib  = numpy.array ( info [ 'channels' ].calibration )
    
    # Gets the raw data information.
    nepoch   = rawdata [ 'epoch_count' ]
    sepoch   = rawdata [ 'epoch_length' ]
    starts   = rawdata [ 'epoch_start' ]
    chorder  = rawdata [ 'channel_order' ]
    rawdata  = rawdata [ 'data' ]
    
    
    # Initializes the list of epochs.
    data     = [];
    
    # Goes through each epoch but the last.
    for eindex in range ( nepoch - 1 ):
        
        # Reads the current epoch.
        datum, _ = raweep.read_block ( rawdata, sepoch, nchan, 8 * starts [ eindex ] )
        
        # Converts the bytes stream into an int32 matrix.
        datum    = numpy.frombuffer (datum, dtype = 'int32' )
        datum    = datum.reshape ( nchan, sepoch ).T
        
        
        # Stores the samples.
        data     = data + [ datum ]
    
    
    # Gets the number of samples in the last epoch.
    srem     = nsamp - sepoch * ( nepoch - 1 )
    
    # Reads the last block.
    datum, _ = raweep.read_block ( rawdata, srem, nchan, 8 * starts [ -1 ] )
    
    # Converts the bytes stream into an int32 matrix.
    datum    = numpy.frombuffer (datum, dtype = 'int32' )
    datum    = datum.reshape ( nchan, srem ).T
    
    # Stores the samples.
    data     = data + [ datum ]
    
    
    # Concatenates all the epochs.
    data     = numpy.concatenate ( data, 0 )
    
    # Reorders the channels and applies the calibration.
    data     = data [ :, chorder ]
    data     = chcalib * data;
    
    
    # Returns the data.
    return data



"""
Code for converting the raw data and header into an MNE object.
"""
def read_mne ( filename ):
    
    
    # Reads the CNT file.
    info     = read_info ( filename )
    data     = read_data ( filename )
    
    
    # Adjusts the scale to SI units (volts).
    scale    = numpy.zeros ( len ( info [ 'channels' ] ) )
    ind_uV   = info [ 'channels' ] [ 'unit' ] == 'uV'
    ind_uV   = ind_uV | ( info [ 'channels' ] [ 'unit' ] == 'µV' )
    ind_V    = info [ 'channels' ] [ 'unit' ] == 'V'
    scale [ ind_uV ] = 1e-6
    scale [ ind_V ] = 1e0
    
    # Applies the scale to the data.
    data     = scale * data
    
    
    # Builds the MNE Raw object.
    mneraw   = mne_tools.build_raw (info, data)
    
    
    # Returns the MNE object.
    return mneraw
