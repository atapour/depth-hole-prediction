import os
import os.path
import argparse
import cv2

def make_dir(path):
    if not os.path.isdir(path):
        os.mkdir(path)

parser = argparse.ArgumentParser('processing Synthia data')
parser.add_argument('--input_root', required=True, type=str, help='point it towards the kitti root.')
parser.add_argument('--output_root', required=True, type=str, help='point it towards where you want to copy everything. The directory structure will be made here.')
args = parser.parse_args()

root_input_path = args.input_root

make_dir(args.output_root)
root_output_path = os.path.join(args.output_root, 'hole_prediction')

make_dir(root_output_path)

make_dir(os.path.join(root_output_path, 'RGB'))
make_dir(os.path.join(root_output_path, 'GT'))

rgb_output_path = os.path.join(root_output_path, 'RGB', '%05d.png')
gt_output_path = os.path.join(root_output_path, 'GT', '%05d.png')

i = 0

main_seqS = os.listdir(root_input_path)
left = 'image_02/data'
right = 'image_03/data'

for main_seq in main_seqS:

    if main_seq.startswith('2011'):

        main_seq_path = os.path.join(root_input_path, main_seq)
        sub_seqs = os.listdir(main_seq_path)
        for sub_seq in sub_seqs:

            sub_seq_path = os.path.join(main_seq_path, sub_seq)
            if os.path.isdir(sub_seq_path):

                left_dir_path = os.path.join(sub_seq_path, 'image_02', 'data')
                right_dir_path = os.path.join(sub_seq_path, 'image_03', 'data')

                if (os.path.isdir(left_dir_path) and os.path.isdir(right_dir_path)):
                    num_left_files = len([name for name in os.listdir(left_dir_path) if os.path.isfile(os.path.join(left_dir_path, name))])
                    num_right_files = len([name for name in os.listdir(right_dir_path) if os.path.isfile(os.path.join(right_dir_path, name))])

                    if num_left_files == num_right_files:

                        lefts = os.listdir(left_dir_path)

                        for name in lefts:

                            left = os.path.join(left_dir_path, name)
                            right = os.path.join(right_dir_path, name)

                            if os.path.isfile(left) and os.path.isfile(right):

                                left_img_color = cv2.imread(left)
                                right_img_gray = cv2.imread(right, cv2.IMREAD_GRAYSCALE)
                                left_img_gray = cv2.imread(left, cv2.IMREAD_GRAYSCALE)

                                stereo = cv2.StereoBM_create(numDisparities=128, blockSize=13)

                                stereo.setTextureThreshold(10)
                                stereo.setMinDisparity(0)
                                stereo.setSpeckleWindowSize(50)
                                stereo.setSpeckleRange(64)

                                disparity = stereo.compute(left_img_gray, right_img_gray)

                                gt = disparity.copy()
                                gt[gt >= 10] = 255
                                gt[gt < 10] = 0

                                print('writing file %s, calculated from %s.' % (i, sub_seq_path))

                                cv2.imwrite(rgb_output_path % (i), left_img_color)
                                cv2.imwrite(gt_output_path % (i), gt)

                                i += 1

print('All finished. %s sequences have been processed.' % i)
