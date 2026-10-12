package net.minecraft.client.gui.screen.inventory;

import com.google.gson.JsonObject;
import net.minecraft.client.Minecraft;
import net.minecraft.inventory.container.Slot;
import net.minecraft.item.Items;
import net.minecraft.item.Item;
import net.minecraft.item.ItemStack;
import net.minecraft.entity.player.ServerPlayerEntity;
import net.minecraft.stats.Stats;
import java.lang.reflect.Field;
import java.util.List;
import net.minecraft.client.gui.recipebook.RecipeBookGui;
import net.minecraft.client.gui.recipebook.RecipeBookPage;
import net.minecraft.client.gui.recipebook.RecipeWidget;
import net.minecraft.util.ResourceLocation;
import net.minecraft.util.registry.Registry;

/** Read-only telemetry. Never moves, grants, crafts, or deletes an item. */
public final class CraftGuiObservation {
    private static Object field(Object object, String name) throws ReflectiveOperationException {
        Field f = object.getClass().getDeclaredField(name);
        f.setAccessible(true);
        return f.get(object);
    }

    /** Inspect displayed buttons only; never invoke recipe placement or unlock APIs. */
    private static void book(JsonObject j, Minecraft mc, ServerPlayerEntity serverPlayer) {
        String[] ids = {"oak_planks", "stick", "crafting_table", "wooden_pickaxe", "oak_slab"};
        j.addProperty("book_error", 0);
        j.addProperty("book_open", 0);
        j.addProperty("book_toggle_x", -1);
        j.addProperty("book_toggle_y", -1);
        j.addProperty("preview_item", -1);
        j.addProperty("preview_count", 0);
        for (int i=1; i<=4; i++) {
            j.addProperty("input_"+i+"_item", -1);
            j.addProperty("input_"+i+"_count", 0);
        }
        j.addProperty("oak_planks_id", Registry.ITEM.getId(Items.OAK_PLANKS));
        j.addProperty("stick_id", Registry.ITEM.getId(Items.STICK));
        j.addProperty("crafting_table_id", Registry.ITEM.getId(Items.CRAFTING_TABLE));
        for (String id : ids) {
            ResourceLocation key = new ResourceLocation("minecraft", id);
            j.addProperty("known_"+id, mc.player != null && mc.player.getRecipeBook().isUnlocked(key) ? 1 : 0);
            j.addProperty("server_known_"+id, serverPlayer != null && serverPlayer.getRecipeBook().isUnlocked(key) ? 1 : 0);
            j.addProperty("book_"+id+"_x", -1);
            j.addProperty("book_"+id+"_y", -1);
            j.addProperty("book_"+id+"_single", 0);
            j.addProperty("book_"+id+"_craftable", 0);
        }
        if (!(mc.currentScreen instanceof InventoryScreen)) return;
        InventoryScreen screen = (InventoryScreen)mc.currentScreen;
        RecipeBookGui book = screen.getRecipeGui();
        j.addProperty("book_open", book.isVisible() ? 1 : 0);
        j.addProperty("book_toggle_x", screen.guiLeft+114);
        j.addProperty("book_toggle_y", screen.height/2-13);
        for (Slot slot : screen.getContainer().inventorySlots) {
            if (slot.slotNumber==0) {
                j.addProperty("preview_item", Registry.ITEM.getId(slot.getStack().getItem()));
                j.addProperty("preview_count", slot.getStack().getCount());
            } else if (slot.slotNumber>=1 && slot.slotNumber<=4) {
                j.addProperty("input_"+slot.slotNumber+"_item", Registry.ITEM.getId(slot.getStack().getItem()));
                j.addProperty("input_"+slot.slotNumber+"_count", slot.getStack().getCount());
            }
        }
        if (!book.isVisible()) return;
        try {
            RecipeBookPage page = (RecipeBookPage)field(book, "recipeBookPage");
            for (RecipeWidget button : (List<RecipeWidget>)field(page, "buttons")) {
                if (!button.visible) continue;
                String id = button.getRecipe().getId().getPath();
                for (String wanted : ids) if (id.equals(wanted)) {
                    j.addProperty("book_"+id+"_x", button.x+12);
                    j.addProperty("book_"+id+"_y", button.y+12);
                    j.addProperty("book_"+id+"_single", button.isOnlyOption() ? 1 : 0);
                    j.addProperty("book_"+id+"_craftable", button.getList().isCraftable(button.getRecipe()) ? 1 : 0);
                }
            }
        } catch (ReflectiveOperationException | RuntimeException error) {
            j.addProperty("book_error", 1);
        }
    }
    public static JsonObject read() {
        Minecraft mc = Minecraft.getInstance();
        JsonObject j = new JsonObject();
        boolean active = mc.currentScreen instanceof InventoryScreen && mc.player != null;
        j.addProperty("inventory_screen", active ? 1 : 0);
        j.addProperty("cursor_count", mc.player == null ? 0 : mc.player.inventory.getItemStack().getCount());
        j.addProperty("cursor_planks", mc.player != null && mc.player.inventory.getItemStack().getItem() == Items.OAK_PLANKS
            ? mc.player.inventory.getItemStack().getCount() : 0);
        double scale = mc.mouseHelper.getScaleFactor();
        j.addProperty("gui_width", mc.getMainWindow().getWidth() / scale);
        j.addProperty("gui_height", mc.getMainWindow().getHeight() / scale);
        j.addProperty("mouse_x", mc.mouseHelper.getMouseX() / scale);
        j.addProperty("mouse_y", mc.mouseHelper.getMouseY() / scale);
        int empty = -1, x = -1, y = -1, grid = 0;
        int resultX = -1, resultY = -1;
        if (active) {
            InventoryScreen screen = (InventoryScreen)mc.currentScreen;
            for (Slot slot : screen.getContainer().inventorySlots) {
                if (slot.slotNumber == 0) {
                    resultX = screen.guiLeft + slot.xPos + 8;
                    resultY = screen.guiTop + slot.yPos + 8;
                }
                // PlayerContainer slots 1..4 are actual inputs; slot 0 is ONLY a preview.
                if (slot.slotNumber >= 1 && slot.slotNumber <= 4 && slot.getStack().getItem() == Items.OAK_PLANKS)
                    grid += slot.getStack().getCount();
                if (empty < 0 && slot.slotNumber >= 9 && slot.slotNumber <= 44 && !slot.getHasStack()) {
                    empty = slot.slotNumber;
                    x = screen.guiLeft + slot.xPos + 8;
                    y = screen.guiTop + slot.yPos + 8;
                }
            }
        }
        j.addProperty("grid_planks", grid);
        j.addProperty("result_x", resultX);
        j.addProperty("result_y", resultY);
        j.addProperty("empty_slot", empty);
        j.addProperty("target_x", x);
        j.addProperty("target_y", y);
        ServerPlayerEntity serverPlayer = mc.player == null || mc.getIntegratedServer() == null ? null
            : mc.getIntegratedServer().getPlayerList().getPlayerByUUID(mc.player.getUniqueID());
        int serverPlanks = 0;
        if (serverPlayer != null) {
            for (ItemStack stack : serverPlayer.inventory.mainInventory)
                if (stack.getItem() == Items.OAK_PLANKS) serverPlanks += stack.getCount();
        }
        j.addProperty("server_available", serverPlayer == null ? 0 : 1);
        j.addProperty("server_inventory_planks", serverPlanks);
        j.addProperty("server_cursor_count", serverPlayer == null ? -1 : serverPlayer.inventory.getItemStack().getCount());
        j.addProperty("server_crafted_planks", serverPlayer == null ? -1 : serverPlayer.getStats().getValue(Stats.ITEM_CRAFTED, Items.OAK_PLANKS));
        Item[] targets = {Items.STICK, Items.CRAFTING_TABLE};
        String[] names = {"sticks", "table"};
        for (int i = 0; i < targets.length; i++) {
            Item target = targets[i];
            int inputCount = 0, inventoryCount = 0;
            if (active) {
                for (Slot slot : ((InventoryScreen)mc.currentScreen).getContainer().inventorySlots)
                    if (slot.slotNumber >= 1 && slot.slotNumber <= 4 && slot.getStack().getItem() == target)
                        inputCount += slot.getStack().getCount();
            }
            if (serverPlayer != null) {
                for (ItemStack stack : serverPlayer.inventory.mainInventory)
                    if (stack.getItem() == target) inventoryCount += stack.getCount();
            }
            j.addProperty("cursor_" + names[i], mc.player != null && mc.player.inventory.getItemStack().getItem() == target
                ? mc.player.inventory.getItemStack().getCount() : 0);
            j.addProperty("grid_" + names[i], inputCount);
            j.addProperty("server_inventory_" + names[i], inventoryCount);
            j.addProperty("server_crafted_" + names[i], serverPlayer == null ? -1 : serverPlayer.getStats().getValue(Stats.ITEM_CRAFTED, target));
        }
        book(j, mc, serverPlayer);
        return j;
    }
}
