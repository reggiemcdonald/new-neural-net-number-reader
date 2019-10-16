package com.reggiemcdonald.neural.feedforward.res;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedList;
import java.util.List;
import java.util.zip.GZIPInputStream;

public class ImageLoader {
    /**
     * Created using
     * https://github.com/ralscha/blog/blob/master/mnist/java/src/main/java/ch/rasc/mnist/MnistReader.java
     * as reference
     */

    public static final Path TEST_IMAGES  = Paths.get ("test_images.gz");
    public static final Path TEST_LABELS  = Paths.get ("test_labels.gz");
    public static final Path TRAIN_IMAGES = Paths.get ("train_images.gz");
    public static final Path TRAIN_LABELS = Paths.get ("train_labels.gz");

    public static final Path EXTENDED_TRAIN_IMAGES = Paths.get ("emnist-train-images.gz");
    public static final Path EXTENDED_TRAIN_LABELS = Paths.get ("emnist-train-labels.gz");
    public static final Path EXTENDED_TEST_IMAGES  = Paths.get ("emnist-test-images.gz" );
    public static final Path EXTENDED_TEST_LABELS  = Paths.get ("emnist-test-labels.gz" );

    public static List<NumberImage> load (Path imgs, Path labels) {
        byte[] decomp_images = ImageLoader.unarchive (imgs);
        byte[] decomp_labels = ImageLoader.unarchive (labels);
        List<NumberImage> numberImages = new LinkedList<>();
        if (decomp_images != null && decomp_labels != null) {
            ByteBuffer imgBuffer   = ByteBuffer.wrap (decomp_images);
            ByteBuffer labelBuffer = ByteBuffer.wrap (decomp_labels);

            if (imgBuffer.getInt() != 2051 || labelBuffer.getInt() != 2049)
                throw new RuntimeException ("Error in opening the zip file");

            int numberOfImages = imgBuffer.getInt();
            int columns        = imgBuffer.getInt();
            int rows           = imgBuffer.getInt();
            labelBuffer.getInt(); // First index is the number of labels

            for (int i = 0; i < numberOfImages; i++) {
                double[][] img = new double[columns][rows];
                for (int x = 0; x < columns; x++) {
                    for (int y = 0; y < rows; y++) {
                        int pixel = imgBuffer.get() & 0xff;
                        img[x][y] = pixel;
                    }
                }
                numberImages.add (new NumberImage(img, labelBuffer.get() & 0xff));
            }

        }
        return numberImages;
    }

    private static byte[] unarchive (Path p) {
        try (
                ByteArrayInputStream stream  = new ByteArrayInputStream (Files.readAllBytes(p));
                GZIPInputStream gzip    = new GZIPInputStream (stream);
                ByteArrayOutputStream bStream = new ByteArrayOutputStream ()
        ) {
            byte[] buffer = new byte[9000];
            int i;
            while ((i = gzip.read (buffer)) > 0)
                bStream.write (buffer, 0,i);

            return bStream.toByteArray();

        } catch (Exception e) {
            System.out.println(e.getMessage());
            return null;
        }
    }

    public static void saveImage (double[][] img) throws IOException {
        BufferedImage bImg = new BufferedImage(24,24, BufferedImage.TYPE_INT_RGB);
        for (int y = 0 ; y < 24; y++) {
            for (int x = 0; x < 24; x++)
                bImg.setRGB(x,y,(int)img[y][x]);
        }
        File out = new File("saved.bmp");
        ImageIO.write (bImg, "png", out);
    }
}
