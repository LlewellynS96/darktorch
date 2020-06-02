def predict(self, dataset, confidence_threshold=0.1, overlap_threshold=0.5, show=True, export=True, gt=False):
    self.eval()

    dataloader = DataLoader(dataset=dataset,
                            batch_size=dataset.batch_size,
                            num_workers=NUM_WORKERS)

    image_idx_ = []
    bboxes_ = []
    confidence_ = []
    classes_ = []

    with torch.no_grad():
        with tqdm(total=len(dataloader),
                  desc='Exporting',
                  leave=True) as pbar:
            for data in dataloader:
                if gt:
                    images, image_info, targets = data
                    targets = targets.to(self.device)
                    bboxes, _, _, image_idx = self.process_bboxes(targets,
                                                                  image_info,
                                                                  0.,
                                                                  1.,
                                                                  nms=False)
                    images_bboxes = []
                    for idx in image_info['id']:
                        mask = np.array(image_idx) == idx
                        images_bboxes.append(bboxes[mask])
                else:
                    images, image_info = data
                images = images.to(self.device)
                predictions = self(images)
                bboxes, classes, confidences, image_idx = self.process_bboxes(predictions,
                                                                              image_info,
                                                                              confidence_threshold,
                                                                              overlap_threshold,
                                                                              nms=True)

                if show:
                    for i, (idx, image) in enumerate(zip(image_info['id'], images)):
                        width = self.image_size[0]
                        height = self.image_size[1]
                        if image.shape[0] == 3:
                            image = to_numpy_image(image, size=(width, height))
                        else:
                            mu = dataset.mu[0]
                            sigma = dataset.sigma[0]
                            image = to_numpy_image(image[0], size=(width, height), mu=mu, sigma=sigma, normalised=False)
                        mask = np.array(image_idx) == idx
                        for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                            name = dataset.classes[cls]
                            if gt:
                                ious = jaccard(bbox[None], images_bboxes[i])
                                max_iou, _ = torch.max(ious, dim=1)
                                if max_iou >= 0.5:
                                    add_bbox_to_image(image, bbox, confidence, name, 2, [0., 255., 0.])
                                else:
                                    add_bbox_to_image(image, bbox, confidence, name, 2, [255., 0., 0.])
                            else:
                                add_bbox_to_image(image, bbox, confidence, name)
                        plt.imshow(image)
                        plt.axis('off')
                        plt.show()

                if export:
                    for idx in range(len(images)):
                        mask = [True if idx_ == image_info['id'][idx] else False for idx_ in image_idx]
                        for bbox, cls, confidence in zip(bboxes[mask], classes[mask], confidences[mask]):
                            name = dataset.classes[cls]
                            ids = image_info['id'][idx]
                            set_name = image_info['dataset'][idx]
                            confidence = confidence.item()
                            bbox[::2] -= image_info['padding'][0][idx]
                            bbox[1::2] -= image_info['padding'][1][idx]
                            bbox[::2] /= image_info['scale'][0][idx]
                            bbox[1::2] /= image_info['scale'][1][idx]
                            x1, y1, x2, y2 = bbox.detach().cpu().numpy()
                            export_prediction(cls=name,
                                              prefix=self.name,
                                              image_id=ids,
                                              left=x1,
                                              top=y1,
                                              right=x2,
                                              bottom=y2,
                                              confidence=confidence,
                                              set_name=set_name)

                bboxes_.append(bboxes)
                confidence_.append(confidences)
                classes_.append(classes)
                image_idx_.append(image_idx)

                pbar.update()

        if len(bboxes_) > 0:
            bboxes = torch.cat(bboxes_).view(-1, 4)
            classes = torch.cat(classes_).flatten()
            confidence = torch.cat(confidence_).flatten()
            image_idx = [item for sublist in image_idx_ for item in sublist]

            return bboxes, classes, confidence, image_idx
        else:
            return torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   torch.tensor([], device=self.device), \
                   []