while getopts ":p:d:h" flag; do
    case "${flag}" in
        p) processes=${OPTARG};;
        d) device=${OPTARG};;\
        h) helpGet && exit;;
        \?) echo "Error: Invalid -$OPTARG option"
            echo
            exit;;
        :) echo "Missing argument for -$OPTARG"
            echo
            exit;;
    esac
done

if [[ -z "$processes" ]] || [[ -z "$device" ]]; then
    processes=1
    device=0
elif [ "$processes" = '' ]; then
    processes=0
elif [ "$device" = '' ]; then
    processes='cpu'
fi

if [ $OPTIND -eq 1 ]; then
    echo
    echo "No options were passed" 
    echo "Run with -h flag for Help"
    echo
    exit;
fi

echo "Processes: $processes";
echo "Device: $device";

dicom_ids=()
for dirname in /input/sub-*; do
    id=${dirname#/input/}
    id=${id%/}
    id=${id%.*}
    dicom_ids+=( "$id" )
done

parallel --jobs ${processes} --progress python3 /app/inference.py sub-23 sub-23_t1_brain-final.nii.gz sub-23_fl_brain-final.nii.gz /input cpu 1 1 ::: ${dicom_ids[@]}

find /tmp -mindepth 1 -delete
echo "Temp files cleared" & wait $!
