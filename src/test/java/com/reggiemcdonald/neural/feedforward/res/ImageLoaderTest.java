package com.reggiemcdonald.neural.feedforward.res;

import org.junit.jupiter.api.Test;

import java.io.File;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

public class ImageLoaderTest {

    @Test
    public void testLoad() {
        List<NumberImage> images = ImageLoader.load(ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS);
        assertNotNull(images);
        assertEquals(10000, images.size());
        for (NumberImage image : images)
            assertNotNull(image);
    }

    @Test
    public void testSaveImage() throws Exception {
        String filename = "test-img";
        File f = new File(filename);
        if (f.exists()) {
            if (f.isDirectory()) {
                fail("a directory exists under the name number.png");
            }
            assertTrue(f.delete());
        }
        List<NumberImage> images = ImageLoader.load(ImageLoader.TEST_IMAGES, ImageLoader.TEST_LABELS);
        ImageLoader.saveImage(images.get(0).image, filename);
        assertTrue((f.exists() && !f.isDirectory()));
        f.delete();
    }
}
