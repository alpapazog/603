function data = loadIDX(filename)
    % Open the file
    fid = fopen(filename, 'rb');
    if fid == -1
        error('Could not open %s. Check if the file exists.', filename);
    end
    
    % Read the magic number (big-endian)
    magicNumber = fread(fid, 1, 'int32', 0, 'ieee-be');
    
    % Determine the type of file based on the magic number
    if magicNumber == 2051
        % Magic number for image file
        numImages = fread(fid, 1, 'int32', 0, 'ieee-be');
        numRows = fread(fid, 1, 'int32', 0, 'ieee-be');
        numCols = fread(fid, 1, 'int32', 0, 'ieee-be');
        data = fread(fid, inf, 'unsigned char');
        data = reshape(data, [numRows, numCols, numImages]);
    elseif magicNumber == 2049
        % Magic number for label file
        numLabels = fread(fid, 1, 'int32', 0, 'ieee-be');
        data = fread(fid, inf, 'unsigned char');
    else
        error('Invalid magic number in file %s.', filename);
    end
    
    fclose(fid);
    data = double(data); % Convert to double for further processing
end
