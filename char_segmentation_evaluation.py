from PIL import Image
import os.path
import xml.etree.ElementTree as ET
import numpy as np
import statistics
from glob import glob
from os.path import splitext, basename
import cv2
import prediction_utils as prediction_utils
from datetime import datetime

class MetricIndicator:

	def __init__(self):
		self.precision = 0
		self.recall = 0
		self.true_positive = 0
		self.false_positive = 0
		self.false_negative = 0
		self.labeled_samples_total = 0


	def precision_recall(self):
		return self.precision_recall_base(self.true_positive, self.false_positive, self.false_negative)

	def precision_recall_base(self, true_positive, false_positive, false_negative):
		precicion = 0
		recall = 0
		if (true_positive + false_positive)!=0:
			precicion = true_positive / (true_positive + false_positive)
		if (true_positive + false_negative) != 0:
			recall = true_positive / (true_positive + false_negative)
		return precicion, recall

	def imprimir_precision_recall(self, precision, recall, class_object):
		print('object class: %s | precision:  %.2f  | recall: %.2f ' % (class_object, precision, recall))

	def imprimir_precision_recall_all(self):
		precision, recall = self.precision_recall()
		self.imprimir_precision_recall(precision, recall, 'total')


	def imprimir_fpn(self, true_positive, false_positive, false_negative, class_object):
		print('object class: %s | true positives: %s | false positives: %s | false negatives: %s  ' % (class_object, true_positive, false_positive, false_negative))

	def imprimir_total_amostras(self):
		print('total amostras: %s | total amostras carros: %s | total amostras moto: %s  ' % (self.labeled_samples_total, self.labeled_samples_total_car, self.labeled_samples_total_moto))

	def imprimir_probabilidades(self):
		if len(self.true_positive_probability)>0 and len(self.false_positive_probability)>0:
			print('True positives:  %f media  %f std  | False positives %f media %f std ' %
				  (statistics.mean(self.true_positive_probability), statistics.stdev(self.true_positive_probability),
				   statistics.mean(self.false_positive_probability), statistics.stdev(self.false_positive_probability)))


	def imprmir_ftn_all(self):
		# self.imprimir_total_amostras()
		print('total de amostras: %i' % self.labeled_samples_total)
		self.imprimir_fpn(self.true_positive, self.false_positive, self.false_negative, 'total')



def calc_iou(gt_bbox, pred_bbox):
	'''
	This function takes the predicted bounding box and ground truth bounding box and
	return the IoU ratio
	'''
	x_topleft_gt, y_topleft_gt, x_bottomright_gt, y_bottomright_gt = gt_bbox
	x_topleft_p, y_topleft_p, x_bottomright_p, y_bottomright_p = pred_bbox
	if (x_topleft_gt > x_bottomright_gt) or (y_topleft_gt > y_bottomright_gt):
		raise AssertionError("Ground Truth Bounding Box is not correct")
	if (x_topleft_p > x_bottomright_p) or (y_topleft_p > y_bottomright_p):
		raise AssertionError("Predicted Bounding Box is not correct", x_topleft_p, x_bottomright_p, y_topleft_p, y_bottomright_gt)
	# if the GT bbox and predcited BBox do not overlap then iou=0
	if x_bottomright_gt < x_topleft_p:
		# If bottom right of x-coordinate  GT  bbox is less than or above the top left of x coordinate of  the predicted BBox
		return 0.0
	if y_bottomright_gt < y_topleft_p:  # If bottom right of y-coordinate  GT  bbox is less than or above the top left of y coordinate of  the predicted BBox
		return 0.0
	if x_topleft_gt > x_bottomright_p:  # If bottom right of x-coordinate  GT  bbox is greater than or below the bottom right  of x coordinate of  the predcited BBox
		return 0.0
	if y_topleft_gt > y_bottomright_p:  # If bottom right of y-coordinate  GT  bbox is greater than or below the bottom right  of y coordinate of  the predcited BBox
		return 0.0
	GT_bbox_area = (x_bottomright_gt - x_topleft_gt + 1) * (y_bottomright_gt - y_topleft_gt + 1)
	Pred_bbox_area = (x_bottomright_p - x_topleft_p + 1) * (y_bottomright_p - y_topleft_p + 1)
	x_top_left = np.max([x_topleft_gt, x_topleft_p])
	y_top_left = np.max([y_topleft_gt, y_topleft_p])
	x_bottom_right = np.min([x_bottomright_gt, x_bottomright_p])
	y_bottom_right = np.min([y_bottomright_gt, y_bottomright_p])
	intersection_area = (x_bottom_right - x_top_left + 1) * (y_bottom_right - y_top_left + 1)
	union_area = (GT_bbox_area + Pred_bbox_area - intersection_area)
	return intersection_area / union_area


'''
predictions_bbox_frame: list of tuples (top_left_x, top_left_y, bottom_right_x, bottom_right_y
ground_truth_frame: list of tuples (top_left_x, top_left_y, bottom_right_x, bottom_right_y
'''
def evaluate_precision_recall(ground_truth_frame: list, predictions_bbox_frame: list, iou_threshold: int = 0.6):
	matched_ground_truth_indexes = []
	true_positive = 0
	for index_pred, pred_bbox in enumerate(predictions_bbox_frame):
		for index_gt, gt_bbox in enumerate(ground_truth_frame):
			iou = calc_iou(gt_bbox, pred_bbox)
			if iou >= iou_threshold and index_gt not in matched_ground_truth_indexes:
				true_positive += 1
				matched_ground_truth_indexes.append(index_gt)
	false_negative = len(ground_truth_frame) - true_positive
	false_positive = len(predictions_bbox_frame) - true_positive
	return true_positive, false_positive, false_negative


def validate_char_segmentation_model(dir_images_validation_input: str, dir_gt_validation_input: str,
									 yolo_char_seg_model: str, dir_images_validation_output: str,
									 is_print_output: bool = False):
	seg_threshold = .5
	imgs_paths = glob('%s/*.jpg' % dir_images_validation_input, recursive=True)
	indicadores_validacao = MetricIndicator()
	tempo_inferencia_list =[]
	tempo_inicial_total = datetime.now()
	for i, img_path in enumerate(imgs_paths):
		try:
			# print('\t Processing %s' % img_path)
			bname_image_file = splitext(basename(img_path))[0]
			name_file_gt = bname_image_file + '.xml'
			plate = cv2.imread(img_path)
			tempo_parcial_inicial = datetime.now()
			predictions = yolo_char_seg_model.return_predict(plate)
			tempo_inferencia_list.append((datetime.now() - tempo_parcial_inicial).total_seconds())
			ground_truth_frame = []
			gt_img_path = os.path.abspath(os.path.join(dir_gt_validation_input, name_file_gt))
			tree = ET.parse(gt_img_path)
			root = tree.getroot()
			for boxes in root.iter('object'):
				ymin, xmin, ymax, xmax = None, None, None, None
				ymin = float(boxes.find("bndbox/ymin").text)
				xmin = float(boxes.find("bndbox/xmin").text)
				ymax = float(boxes.find("bndbox/ymax").text)
				xmax = float(boxes.find("bndbox/xmax").text)
				ground_truth_frame.append((xmin, ymin, xmax, ymax))
				indicadores_validacao.labeled_samples_total +=1
			predictions_bbox_frame = []
			for prediction in predictions:
				if prediction.get("confidence") > 0.10:
					xtop = prediction.get('topleft').get('x')
					ytop = prediction.get('topleft').get('y')
					xbottom = prediction.get('bottomright').get('x')
					ybottom = prediction.get('bottomright').get('y')
					predictions_bbox_frame.append((xtop, ytop, xbottom, ybottom))
					# char = img[ytop:ybottom, xtop:xbottom]
					# cv2.rectangle(img, (xtop, ytop), (xbottom, ybottom), (255, 0, 0), 2)
			true_positive_frame, false_positive_frame, false_negative_frame = evaluate_precision_recall(ground_truth_frame, predictions_bbox_frame, seg_threshold)
			if false_negative_frame > 0:
				print('false negative: %s ' % img_path)
			if is_print_output:
				plate_copy = plate.copy()
				for bbox_predicted in predictions_bbox_frame:
					xtop, ytop, xbottom, ybottom = bbox_predicted
					cv2.rectangle(plate_copy, (xtop, ytop), (xbottom, ybottom), (255, 0, 0), 2)
				image_pil = Image.fromarray(plate_copy)
				output_name_file = bname_image_file + '.jpg'
				abs_path_file_output = os.path.abspath(os.path.join(dir_images_validation_output, output_name_file))
				image_pil.save(abs_path_file_output)
			indicadores_validacao.true_positive += true_positive_frame
			indicadores_validacao.false_positive += false_positive_frame
			indicadores_validacao.false_negative += false_negative_frame
		except Exception as error:
			print('error validating image %s'  % img_path)
			print(error)
	tempo_total_final = datetime.now()
	print('tempo total: {}s | media: {}s | min: {}s | max: {}s '.format(
		(tempo_total_final-tempo_inicial_total).total_seconds(), statistics.mean(tempo_inferencia_list),
		min(tempo_inferencia_list), max(tempo_inferencia_list)))
	indicadores_validacao.imprmir_ftn_all()
	indicadores_validacao.imprimir_precision_recall_all()